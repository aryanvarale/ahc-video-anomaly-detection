"""
Evaluate several LoRA checkpoints against the held-out set and print a
comparison, so we ship the checkpoint that actually scores best rather than
assuming the last epoch is the best one.

The previous run was already at ~98% token accuracy by epoch 2, so a later
epoch can easily be overfitting rather than improving - the only way to know is
to score the checkpoints against held-out data.

For each checkpoint: merge LoRA -> serve on vLLM -> cache per-chunk verdicts for
all 34 videos -> aggregate into events -> score. Each one costs ~8.3 GB of disk
while it is being evaluated and is removed afterwards unless --keep is passed.

  python eval_checkpoints.py --ckpts out3/v1-*/checkpoint-900 out3/v1-*/checkpoint-2499
  python eval_checkpoints.py --auto            # last + two earlier, spread out

Refuses to start while a training job is running, because serving a model
alongside it risks OOM-killing the training.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

import vad_pipeline as vp
from scorer import load_gt as load_gt_levels, score_l1, score_l23
from metrics import load_gt as load_gt_events, score as score_events

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
VIDEOS = ROOT / "data/Train and Test/test/videos"
GT = ROOT / "data/Train and Test/test/ground_truth.csv"
PORT = int(os.environ.get("EVAL_PORT", "28461"))
NVRTC = "/home/miniorange/.local/lib/python3.10/site-packages/nvidia/cu13/lib"


def training_is_running():
    """True only for an actual `python .../swift/cli/sft.py` process.

    Shell wrappers that merely mention the path - a `pgrep`/`until` loop waiting
    on training, or this script's own invocation - must not count, or the guard
    matches itself and blocks a run that is perfectly safe."""
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if "swift/cli/sft.py" not in ln:
            continue
        if any(tok in ln for tok in ("pgrep", "grep ", "until ", "eval_checkpoints", "/bin/bash")):
            continue
        if "python" in ln:
            return True
    return False


def sh(cmd, **kw):
    env = {**os.environ, "LD_LIBRARY_PATH": f"{NVRTC}:{os.environ.get('LD_LIBRARY_PATH','')}"}
    return subprocess.run(cmd, shell=True, env=env, **kw)


def merge(ckpt: Path) -> Path:
    merged = ckpt.parent / f"{ckpt.name}-merged"
    if merged.exists():
        print(f"    merged already present: {merged.name}")
        return merged
    print(f"    merging LoRA -> {merged.name} ...")
    r = sh(f'swift export --adapters "{ckpt}" --merge_lora true',
           capture_output=True, text=True)
    if not merged.exists():
        print(r.stdout[-1500:] or "", r.stderr[-1500:] or "")
        raise RuntimeError(f"merge failed for {ckpt}")
    return merged


def serve(merged: Path):
    print(f"    serving on :{PORT} ...")
    env = {**os.environ,
           "LD_LIBRARY_PATH": f"{NVRTC}:{os.environ.get('LD_LIBRARY_PATH','')}",
           "MAX_PIXELS": "200704", "VIDEO_MAX_PIXELS": "200704",
           "CUDA_VISIBLE_DEVICES": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    log = open(ROOT / "eval_vllm.log", "w")
    proc = subprocess.Popen(
        ["vllm", "serve", str(merged), "--served-model-name", vp.CFG["model"],
         "--port", str(PORT), "--max-model-len", "8192",
         "--limit-mm-per-prompt", '{"image":16}', "--dtype", "bfloat16",
         "--gpu-memory-utilization", "0.40"],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    for _ in range(600):                       # up to ~10 min for load + warmup
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited during startup - see eval_vllm.log")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).ok:
                print("    server ready")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("vLLM did not become ready in time")


def stop(proc):
    if not proc:
        return
    try:
        os.killpg(os.getpgid(proc.pid), 15)
    except Exception:
        proc.kill()
    for _ in range(30):
        if proc.poll() is not None:
            break
        time.sleep(1)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            pass
    time.sleep(3)                              # let VRAM actually come back


def build_preds(cache):
    """Cached per-chunk distributions -> the same events the pipeline would emit."""
    preds = {}
    for vid, e in cache.items():
        chunks = [(w[0], w[1], None) for w in e["windows"]]
        probs = e["probs"]
        if e["level"] == 1:
            agg = {}
            for p in probs:
                for c, v in p.items():
                    agg[c] = agg.get(c, 0.0) + v
            z = sum(agg.values()) or 1.0
            agg = {k: v / z for k, v in agg.items()}
            if 1.0 - agg.get("normal", 0.0) < vp.CFG["t_level1"]:
                events = []
            else:
                cls = max((c for c in agg if c != "normal"), key=lambda c: agg[c])
                events = [{"class_name": cls, "start_time_sec": None,
                           "end_time_sec": None, "confidence": round(agg[cls], 4),
                           "explanation": vp.EXPL.get(cls, "")}]
        else:
            events = [{"class_name": c, "start_time_sec": s, "end_time_sec": en,
                       "confidence": round(sc, 4), "explanation": vp.EXPL.get(c, "")}
                      for c, s, en, sc in vp.aggregate(chunks, probs)]
        preds[vid] = {"video_id": vid, "events": events}
    return preds


def val_accuracy(val_path, limit):
    """Per-class accuracy on held-out *videos*.

    The 34-video arena-style set is too small to separate two checkpoints (4 of
    them are level 3), so the checkpoint decision leans on this instead: ~1000
    chunks from videos no training step saw."""
    out = ROOT / "val_tmp.json"
    r = subprocess.run(
        [sys.executable, "eval_val.py", "--val", val_path, "--limit", str(limit),
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "VAD_SERVER": f"http://127.0.0.1:{PORT}/v1/chat/completions"})
    if not out.exists():
        print("    val eval failed:", r.stdout[-400:], r.stderr[-400:])
        return None
    d = json.load(open(out))
    out.unlink(missing_ok=True)
    return d


def evaluate(name, merged, keep_cache):
    cache_path = ROOT / f"cache_{name}.json"
    if not cache_path.exists():
        print("    running 34 videos ...")
        env = {**os.environ, "VAD_SERVER": f"http://127.0.0.1:{PORT}/v1/chat/completions"}
        subprocess.run([sys.executable, "cache_chunks.py", "--manifest", str(MANIFEST),
                        "--videos", str(VIDEOS), "--out", str(cache_path)],
                       cwd=ROOT, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    cache = json.load(open(cache_path))
    preds = build_preds(cache)

    lg, eg = load_gt_levels(str(GT)), load_gt_events(str(GT))
    r1, r2, r3 = score_l1(preds, lg), score_l23(preds, lg, 2), score_l23(preds, lg, 3)
    ev = score_events(preds, eg)

    if not keep_cache:
        cache_path.unlink(missing_ok=True)

    s1 = r1["score"] if r1 else 0.0
    s2 = r2["mean_score"] if r2 else 0.0
    s3 = r3["mean_score"] if r3 else 0.0
    return {"name": name, "L1": s1, "L2": s2, "L3": s3, "val": None,
            # weighted the way the arena splits its marks (25 / 35 / 40)
            "weighted": 0.25 * s1 + 0.35 * s2 + 0.40 * s3,
            "precision": ev["precision"], "recall": ev["recall"], "f1": ev["f1"],
            "fp": ev["fp"], "preds": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--auto", action="store_true",
                    help="pick the final checkpoint plus two earlier ones")
    ap.add_argument("--run-dir", default="out3")
    ap.add_argument("--keep", action="store_true", help="keep merged weights + caches")
    ap.add_argument("--force", action="store_true", help="run even while training")
    ap.add_argument("--val", default="", help="held-out SFT jsonl for per-class accuracy")
    ap.add_argument("--val_limit", type=int, default=1100)
    a = ap.parse_args()

    if training_is_running() and not a.force:
        sys.exit("Training is still running - serving a model now risks OOM-killing it.\n"
                 "Wait for it to finish, or pass --force if you really mean to.")

    ckpts = [Path(c) for c in a.ckpts]
    if a.auto or not ckpts:
        found = sorted(Path(a.run_dir).glob("v*/checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        found = [p for p in found if not p.name.endswith("-merged")]
        if not found:
            sys.exit(f"no checkpoints found under {a.run_dir}/")
        ckpts = sorted({found[-1], found[len(found) // 2], found[max(len(found) - 4, 0)]},
                       key=lambda p: int(p.name.split("-")[1]))

    print(f"\nevaluating {len(ckpts)} checkpoint(s): {[c.name for c in ckpts]}\n")
    results = []
    for ckpt in ckpts:
        name = f"{ckpt.parent.name}_{ckpt.name}"
        print(f"[{ckpt.name}]")
        proc = merged = None
        try:
            merged = merge(ckpt)
            proc = serve(merged)
            res = evaluate(name, merged, a.keep)
            if a.val:
                res["val"] = val_accuracy(a.val, a.val_limit)
            results.append(res)
            print(f"    L1 {res['L1']:.3f}  L2 {res['L2']:.3f}  L3 {res['L3']:.3f}"
                  f"   P {res['precision']*100:.1f}%  R {res['recall']*100:.1f}%"
                  f"  F1 {res['f1']*100:.1f}%  FP {res['fp']}")
            if res["val"]:
                pc = res["val"]["per_class"]
                worst = sorted(pc.items(), key=lambda x: x[1])[:4]
                print(f"    val {res['val']['overall']*100:.1f}%  weakest: "
                      + ", ".join(f"{k} {v*100:.0f}%" for k, v in worst))
            print()
        except Exception as exc:                                  # noqa: BLE001
            print(f"    FAILED: {exc}\n")
        finally:
            stop(proc)
            if merged and merged.exists() and not a.keep:
                shutil.rmtree(merged, ignore_errors=True)

    if not results:
        sys.exit("nothing evaluated successfully")

    print("=" * 78)
    print(f"{'checkpoint':<26}{'L1':>7}{'L2':>7}{'L3':>7}{'wtd':>8}{'prec':>8}{'rec':>7}{'FP':>5}{'val':>7}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: -x["weighted"]):
        v = f"{r['val']['overall']*100:>6.1f}%" if r.get("val") else "     -"
        print(f"{r['name']:<26}{r['L1']:>7.3f}{r['L2']:>7.3f}{r['L3']:>7.3f}"
              f"{r['weighted']:>8.3f}{r['precision']*100:>7.1f}%{r['recall']*100:>6.1f}%{r['fp']:>5}{v}")
    print("=" * 85)

    best = max(results, key=lambda x: x["weighted"])
    print(f"\nbest by weighted score: {best['name']}")
    out = ROOT / "best_checkpoint_preds.json"
    json.dump({"checkpoint": best["name"],
               "predictions": list(best["preds"].values())}, open(out, "w"), indent=1)
    print(f"its predictions -> {out.name}"
          "\n(build a submission from these once you've picked a winner)")


if __name__ == "__main__":
    main()
