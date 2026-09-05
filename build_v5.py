"""
v5 = v4 with the level-3 videos re-run at the training chunk geometry.

Only level 3 changes. Level 1 keeps v4's dense-chunk predictions (the path that
took D1 from 5/20 to 12/20, which vad_pipeline.main() does not implement) and
level 2 was already running at 2.0s, so re-running either would only add noise.

The level-3 videos are genuinely re-run rather than rebuilt from a cached
probability dump, because runtime_metadata has to describe calls that actually
happened - a previous submission reused stale metadata claiming one call per
video when dense chunking makes hundreds.
"""
import json
import time

from concurrent.futures import ThreadPoolExecutor

import vad_pipeline as vp
from scorer import load_gt, score_l1, score_l23

V4 = "submission_ft_v4.json"
OUT = "submission_ft_v5.json"
VIDEOS = "data/Train and Test/test/videos"
GT = "data/Train and Test/test/ground_truth.csv"

v4 = json.load(open(V4))
levels = {}
for r in json.load(open("manifest.json"))["videos"]:
    levels[r["video_id"]] = int(r.get("level", 1))

l3 = sorted(v for v, l in levels.items() if l == 3)
print(f"re-running level 3 at {vp.CFG['chunk_sec'][3]}s/{vp.CFG['stride_sec'][3]}s: {l3}")

t0 = time.perf_counter()
fresh = {}
with ThreadPoolExecutor(vp.CFG["concurrency"]) as pool:
    for vid in l3:
        p = vp.do_video(vid, f"{VIDEOS}/{vid}.mp4", 3, pool)
        fresh[vid] = p
        mr = p["runtime_metadata"]["model_runtimes"][0]
        print(f"  {vid}: {len(p['events'])} events, {mr['call_count']} calls, "
              f"{p['runtime_metadata']['end_to_end_internal_time_ms']:.0f} ms")
l3_ms = (time.perf_counter() - t0) * 1000.0

preds = [fresh.get(p["video_id"], p) for p in v4["predictions"]]

# v4's own level-1/2 wall time still stands; add what level 3 actually cost now.
old_l3_ms = sum(p["runtime_metadata"]["end_to_end_internal_time_ms"]
                for p in v4["predictions"] if levels.get(p["video_id"]) == 3)
sub = {
    "schema_version": "1.0",
    "submission_id": "run-ft5-l3-train-geometry",
    "model_name": "qwen3vl4b-lora-e3-dense-tempered",
    "run_metadata": {
        "total_wall_time_ms": round(
            v4["run_metadata"]["total_wall_time_ms"] - old_l3_ms + l3_ms, 1),
        "max_parallel_videos": 1,
        "hardware": v4["run_metadata"]["hardware"],
    },
    "predictions": preds,
}

errs = vp.validate(preds, levels)
if errs:
    print("\nVALIDATION FAILED:")
    for e in errs[:40]:
        print(" -", e)
    raise SystemExit(1)

json.dump(sub, open(OUT, "w"), indent=1)

gt = load_gt(GT)
pm = {p["video_id"]: p for p in preds}
r1, r2, r3 = score_l1(pm, gt), score_l23(pm, gt, 2), score_l23(pm, gt, 3)
marks = 25 * r1["score"] + 35 * r2["mean_score"] + 40 * r3["mean_score"]
print(f"\nv5  L1 {r1['score']:.3f}  L2 {r2['mean_score']:.3f}  L3 {r3['mean_score']:.3f}"
      f"   est {marks:.1f}/100")
print("   L3 per video:", {k: round(v, 2) for k, v in r3["per_video"].items()})
print(f"wrote {OUT}")
