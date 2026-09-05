"""
Joint sweep over the aggregation thresholds, against the 34 ground-truthed videos.

Level 3 is the gap on the live board - 14.0 against 19.7-22.9 for everyone above -
and the thresholds are committed config, so anything found here is reproducible
model output rather than a hand edit.

Discipline, because tuning on this set has cost real marks before (a submission
tuned on the aggregate weighted score dropped the arena score 40.5 -> 38.4):

 * report every level separately, never a single weighted number, since the set is
   34 videos with only 4 at level 3 and an aggregate win is routinely one video
   flipping;
 * prefer a setting sitting in the middle of a PLATEAU of good neighbours over an
   isolated peak, which is almost always noise;
 * report how many neighbouring settings score within a hair of the winner, so a
   lucky spike is visible as a lucky spike.

  python sweep_joint.py --cache_path cache_prod_raw.json
"""
import argparse
import itertools
import json

import vad_pipeline as vp
from scorer import load_gt, score_l1, score_l23

GT = "data/Train and Test/test/ground_truth.csv"


def evaluate(cache, gt, T, t_high, t_low, t_video, max_events):
    vp.CFG["t_high"], vp.CFG["t_low"] = t_high, t_low
    vp.CFG["t_video"], vp.CFG["max_events"] = t_video, max_events
    preds = {}
    for vid, e in cache.items():
        probs = [vp.apply_temperature(p, T) for p in e["probs"]]
        chunks = [(w[0], w[1], None) for w in e["windows"]]
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
                           "end_time_sec": None, "explanation": vp.EXPL[cls]}]
        else:
            events = [{"class_name": c, "start_time_sec": s, "end_time_sec": en,
                       "explanation": vp.EXPL[c]}
                      for c, s, en, _ in vp.aggregate(chunks, probs)]
        preds[vid] = {"video_id": vid, "events": events}

    r1, r2, r3 = score_l1(preds, gt), score_l23(preds, gt, 2), score_l23(preds, gt, 3)
    return (r1["score"], r2["mean_score"], r3["mean_score"],
            25 * r1["score"] + 35 * r2["mean_score"] + 40 * r3["mean_score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", default="cache_prod_raw.json")
    a = ap.parse_args()

    cache = json.load(open(a.cache_path))
    gt = load_gt(GT)

    base = evaluate(cache, gt, vp.CFG["temperature"], 0.75, 0.40, 0.65, 4)
    print(f"shipped config  L1 {base[0]:.3f}  L2 {base[1]:.3f}  L3 {base[2]:.3f}"
          f"   est {base[3]:.1f}\n")

    grid = list(itertools.product(
        [1.5, 2.0, 2.5, 3.0],            # temperature
        [0.60, 0.70, 0.75, 0.85, 0.92],  # t_high
        [0.25, 0.40, 0.55, 0.70],        # t_low
        [0.55, 0.65, 0.75],              # t_video
        [3, 4],                          # max_events
    ))
    rows = []
    for T, th, tl, tv, me in grid:
        if tl >= th:
            continue
        s1, s2, s3, m = evaluate(cache, gt, T, th, tl, tv, me)
        rows.append({"T": T, "t_high": th, "t_low": tl, "t_video": tv,
                     "max_events": me, "L1": s1, "L2": s2, "L3": s3, "marks": m})

    rows.sort(key=lambda r: -r["marks"])
    print(f"{len(rows)} configs. top 12 by est marks:")
    print(f"{'T':>5}{'t_hi':>6}{'t_lo':>6}{'t_vid':>6}{'max':>5}"
          f"{'L1':>7}{'L2':>7}{'L3':>7}{'marks':>8}")
    for r in rows[:12]:
        print(f"{r['T']:>5}{r['t_high']:>6}{r['t_low']:>6}{r['t_video']:>6}"
              f"{r['max_events']:>5}{r['L1']:>7.3f}{r['L2']:>7.3f}"
              f"{r['L3']:>7.3f}{r['marks']:>8.1f}")

    best = rows[0]
    near = [r for r in rows if r["marks"] >= best["marks"] - 0.75]
    print(f"\nbest {best['marks']:.1f};  {len(near)} configs within 0.75 marks of it")
    if len(near) < 4:
        print("  -> ISOLATED PEAK. Almost certainly noise on a 34-video set; do not ship.")
    else:
        # a parameter that is not shared across the near-optimal set is not doing work
        for k in ("T", "t_high", "t_low", "t_video", "max_events"):
            vals = sorted({r[k] for r in near})
            print(f"  {k:11s} across the good set: {vals}")

    print("\nbest L3 specifically (the live gap):")
    for r in sorted(rows, key=lambda r: -r["L3"])[:5]:
        print(f"  T={r['T']} t_hi={r['t_high']} t_lo={r['t_low']} t_vid={r['t_video']}"
              f" max={r['max_events']}  ->  L1 {r['L1']:.3f} L2 {r['L2']:.3f}"
              f" L3 {r['L3']:.3f}  est {r['marks']:.1f}")


if __name__ == "__main__":
    main()
