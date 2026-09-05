"""
Cache per-chunk probabilities for the served checkpoint, then sweep temperature
offline against the ground-truthed 34-video set.

The rebalanced checkpoint scored 41.4 against the previous model's 49.2, but its
held-out classification is BETTER (traffic_accident magnet 18 -> 9 wrong chunks,
wrong_way 57% -> 63%, overall 93.2% -> 93.5%). Classification improving while the
submission score collapses is a calibration signature, not a worse model: the
temperature and hysteresis thresholds were fitted to the old checkpoint's
probability distribution, and temperature alone has been worth +0.14 on L3 here.

So this caches at temperature 1.0 - raw, untempered - and applies each candidate
temperature offline, which makes the sweep free after one pass over the videos
instead of one pass per setting.

  python tune_checkpoint.py --cache          # one pass over the 34 videos
  python tune_checkpoint.py --sweep          # offline, seconds
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import vad_pipeline as vp
from scorer import load_gt, score_l1, score_l23

VIDEOS = "data/Train and Test/test/videos"
GT = "data/Train and Test/test/ground_truth.csv"
CACHE = "cache_out5_raw.json"


def build_cache(path_out):
    levels = {r["video_id"]: int(r.get("level", 1))
              for r in json.load(open("manifest.json"))["videos"]}
    vp.CFG["temperature"] = 1.0            # cache raw; temperature applied offline
    out = {}
    with ThreadPoolExecutor(vp.CFG["concurrency"]) as pool:
        for i, vid in enumerate(sorted(levels), 1):
            lvl = levels[vid]
            chunks, dur, nf = vp.read_chunks(f"{VIDEOS}/{vid}.mp4", lvl)
            probs = list(pool.map(lambda c: vp.classify(c[2]), chunks))
            out[vid] = {"level": lvl, "duration": dur,
                        "windows": [[c[0], c[1]] for c in chunks], "probs": probs}
            print(f"[{i}/{len(levels)}] {vid} L{lvl} {len(chunks)} chunks", flush=True)
    json.dump(out, open(path_out, "w"))
    print("wrote", path_out)


def preds_at(cache, T):
    """Rebuild the whole submission's events from cached raw probabilities."""
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
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cache_path", default=CACHE)
    a = ap.parse_args()

    if a.cache:
        build_cache(a.cache_path)
        return

    cache = json.load(open(a.cache_path))
    gt = load_gt(GT)
    print(f"{'T':>6} {'L1':>7} {'L2':>7} {'L3':>7} {'est marks':>10}")
    rows = []
    for T in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        p = preds_at(cache, T)
        r1, r2, r3 = score_l1(p, gt), score_l23(p, gt, 2), score_l23(p, gt, 3)
        marks = 25 * r1["score"] + 35 * r2["mean_score"] + 40 * r3["mean_score"]
        rows.append((T, marks))
        print(f"{T:>6} {r1['score']:>7.3f} {r2['mean_score']:>7.3f} "
              f"{r3['mean_score']:>7.3f} {marks:>10.1f}")
    best = max(rows, key=lambda x: x[1])
    print(f"\nbest T={best[0]}  ->  {best[1]:.1f}/100")
    print("previous checkpoint at its own tuned settings: 49.2/100 (scored 51 live)")


if __name__ == "__main__":
    main()
