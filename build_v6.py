"""
Full 34-video submission from whichever model is being served, using v4's
configuration exactly.

v4's level-1 pass was built ad hoc and never committed, so it is reconstructed
here from its outputs: dense 2s/1s chunks (rather than one whole-clip call) and
t_level1=0.3. That reproduces v4's emit-and-class decision on 23 of 24 level-1
videos, and it is just do_video()'s existing sum-and-normalise path with
different chunk settings - so nothing new has to be trusted, and runtime_metadata
describes calls that actually happened.

  VAD_SERVER=... python build_v6.py --out submission_ft_v6.json
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import vad_pipeline as vp
from scorer import load_gt, score_l1, score_l23
from metrics import load_gt as load_gt_events, score as score_events

VIDEOS = "data/Train and Test/test/videos"
GT = "data/Train and Test/test/ground_truth.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="submission_ft_v6.json")
ap.add_argument("--model_name", default="qwen3vl4b-lora-rebalanced")
ap.add_argument("--tag", default="run-ft6-rebalanced")
a = ap.parse_args()

# v4's level-1 geometry. Level 2/3 already match what v4 shipped.
vp.CFG["chunk_sec"][1] = 2.0
vp.CFG["stride_sec"][1] = 1.0
vp.CFG["frames_per_chunk"][1] = 8
vp.CFG["t_level1"] = 0.3

levels = {r["video_id"]: int(r.get("level", 1))
          for r in json.load(open("manifest.json"))["videos"]}

preds, t0 = [], time.perf_counter()
with ThreadPoolExecutor(vp.CFG["concurrency"]) as pool:
    for i, vid in enumerate(sorted(levels), 1):
        p = vp.do_video(vid, f"{VIDEOS}/{vid}.mp4", levels[vid], pool)
        preds.append(p)
        print(f"[{i}/{len(levels)}] {vid} L{levels[vid]} -> "
              f"{[e['class_name'] for e in p['events']] or 'normal'}", flush=True)

sub = {
    "schema_version": "1.0",
    "submission_id": a.tag,
    "model_name": a.model_name,
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

row = {"submission": a.out, "checkpoint": a.tag,
       "L1": round(r1["score"], 4), "anomaly_acc": round(r1["anomaly_acc"], 4),
       "class_acc": round(r1["class_acc"], 4),
       "L2": round(r2["mean_score"], 4), "L3": round(r3["mean_score"], 4),
       "est_marks": round(marks, 1),
       "precision": round(ev["precision"], 4), "recall": round(ev["recall"], 4),
       "f1": round(ev["f1"], 4), "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"]}

print(f"\n{a.out}")
print(f"  L1 {r1['score']:.3f} (anomaly-acc {r1['anomaly_acc']:.3f}  class-acc {r1['class_acc']:.3f})"
      f"  L2 {r2['mean_score']:.3f}  L3 {r3['mean_score']:.3f}")
print(f"  precision {ev['precision']*100:.1f}%  recall {ev['recall']*100:.1f}%  "
      f"F1 {ev['f1']*100:.1f}%  (tp {ev['tp']} fp {ev['fp']} fn {ev['fn']})")
print(f"  est {marks:.1f}/100")
print("  v4 baseline:  L1 0.750 (class-acc 0.583)  L2 0.624  L3 0.216  "
      "P 44.2%  R 41.3%  est 49.2  [scored 51 live]")

# one appendable table across checkpoints, so they can be compared at a glance
with open("checkpoint_results.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
