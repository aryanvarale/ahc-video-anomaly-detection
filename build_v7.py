"""
Submission using shot-segmented detection at levels 2/3.

Level 1 is unchanged (v4's dense 2s/1s pass). Levels 2/3 segment on shot cuts and
classify each shot as a unit; a video with no usable cuts falls back to the
existing sliding-window + hysteresis path, so continuous videos behave exactly as
before.

  VAD_SERVER=... python build_v7.py --out submission_ft_v7.json --t_seg 0.5
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import vad_pipeline as vp
import shot_pipeline as sp
from scorer import load_gt, score_l1, score_l23
from metrics import load_gt as load_gt_events, score as score_events

VIDEOS = "data/Train and Test/test/videos"
GT = "data/Train and Test/test/ground_truth.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="submission_ft_v7.json")
ap.add_argument("--tag", default="run-ft7-shots")
ap.add_argument("--model_name", default="qwen3vl4b-lora-shotseg")
ap.add_argument("--t_seg", type=float, default=0.5,
                help="a shot becomes an event when 1 - p(normal) reaches this")
ap.add_argument("--min_cuts", type=int, default=2,
                help="below this many cuts the video is treated as continuous")
ap.add_argument("--windows_per_shot", type=int, default=6)
a = ap.parse_args()

vp.CFG["chunk_sec"][1] = 2.0
vp.CFG["stride_sec"][1] = 1.0
vp.CFG["frames_per_chunk"][1] = 8
vp.CFG["t_level1"] = 0.3

levels = {r["video_id"]: int(r.get("level", 1))
          for r in json.load(open("manifest.json"))["videos"]}


def shot_video(vid, path, pool):
    t0 = time.perf_counter()
    segs, dur = sp.segments(path)
    if len(segs) - 1 < a.min_cuts:
        return None                              # continuous: use the old path

    probs, times = sp.classify_shots(path, segs, pool, a.windows_per_shot)

    raw = []
    for (s, e), pr in zip(segs, probs):
        if not pr:
            continue
        anom = {c: v for c, v in pr.items() if c != "normal"}
        if not anom or 1.0 - pr.get("normal", 0.0) < a.t_seg:
            continue
        cls = max(anom, key=anom.get)
        raw.append([cls, s, e, anom[cls]])

    # neighbouring shots of the same class are one event, not two
    merged = []
    for ev in raw:
        if merged and merged[-1][0] == ev[0] and ev[1] - merged[-1][2] <= 0.75:
            merged[-1][2] = ev[2]
            merged[-1][3] = max(merged[-1][3], ev[3])
        else:
            merged.append(ev)
    merged.sort(key=lambda x: -x[3])
    merged = sorted(merged[: vp.CFG["max_events"]], key=lambda x: x[1])

    events = [{"class_name": c, "start_time_sec": round(s, 2),
               "end_time_sec": round(e, 2), "confidence": round(sc, 4),
               "explanation": vp.EXPL[c]} for c, s, e, sc in merged]

    times.sort()
    n = max(len(times), 1)
    tot = sum(times)
    return {
        "video_id": vid, "events": events,
        "runtime_metadata": {
            "frames_processed": int(sum(1 for _ in times)) * 8,
            "chunks_processed": len(segs),
            "end_to_end_internal_time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model_runtimes": [{
                "model_name": "qwen3-vl-4b-instruct-lora",
                "call_count": n, "total_time_ms": round(tot, 1),
                "average_time_ms": round(tot / n, 3),
                "p50_time_ms": round(vp._percentile(times, 0.50), 1),
                "p95_time_ms": round(vp._percentile(times, 0.95), 1),
                "max_time_ms": round(max(times), 1) if times else 0.0,
            }],
        },
    }


preds, t0, n_shot = [], time.perf_counter(), 0
with ThreadPoolExecutor(vp.CFG["concurrency"]) as pool:
    for i, vid in enumerate(sorted(levels), 1):
        path = f"{VIDEOS}/{vid}.mp4"
        lvl = levels[vid]
        p = shot_video(vid, path, pool) if lvl != 1 else None
        mode = "shots" if p else ("L1" if lvl == 1 else "continuous")
        if p:
            n_shot += 1
        else:
            p = vp.do_video(vid, path, lvl, pool)
        preds.append(p)
        print(f"[{i}/{len(levels)}] {vid} L{lvl} ({mode}) -> "
              f"{[e['class_name'] for e in p['events']] or 'normal'}", flush=True)

sub = {
    "schema_version": "1.0", "submission_id": a.tag, "model_name": a.model_name,
    "run_metadata": {
        "total_wall_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "max_parallel_videos": 1,
        "hardware": os.environ.get("VAD_HW", "1x RTX PRO 6000"),
    },
    "predictions": preds,
}

errs = vp.validate(preds, levels)
if errs:
    print("\nVALIDATION FAILED:")
    for e in errs[:40]:
        print(" -", e)
    raise SystemExit(1)
json.dump(sub, open(a.out, "w"), indent=1)

gt = load_gt(GT)
pm = {p["video_id"]: p for p in preds}
r1, r2, r3 = score_l1(pm, gt), score_l23(pm, gt, 2), score_l23(pm, gt, 3)
ev = score_events(pm, load_gt_events(GT))
marks = 25 * r1["score"] + 35 * r2["mean_score"] + 40 * r3["mean_score"]

print(f"\n{a.out}   ({n_shot} videos used shot segmentation, t_seg={a.t_seg})")
print(f"  L1 {r1['score']:.3f} (class-acc {r1['class_acc']:.3f})  "
      f"L2 {r2['mean_score']:.3f}  L3 {r3['mean_score']:.3f}")
print(f"  precision {ev['precision']*100:.1f}%  recall {ev['recall']*100:.1f}%  "
      f"F1 {ev['f1']*100:.1f}%   est {marks:.1f}/100")
print(f"  L2 per video: { {k: round(v,3) for k,v in r2['per_video'].items()} }")
print(f"  L3 per video: { {k: round(v,3) for k,v in r3['per_video'].items()} }")
print("  v4 baseline:  L1 0.750  L2 0.624  L3 0.216  P 44.2% R 41.3%  est 49.2 [51 live]")

with open("checkpoint_results.jsonl", "a") as f:
    f.write(json.dumps({"submission": a.out, "checkpoint": a.tag,
                        "L1": round(r1["score"], 4), "class_acc": round(r1["class_acc"], 4),
                        "L2": round(r2["mean_score"], 4), "L3": round(r3["mean_score"], 4),
                        "precision": round(ev["precision"], 4),
                        "recall": round(ev["recall"], 4), "f1": round(ev["f1"], 4),
                        "est_marks": round(marks, 1), "t_seg": a.t_seg}) + "\n")
