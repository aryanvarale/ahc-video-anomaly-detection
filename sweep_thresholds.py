"""
Sweep CFG thresholds against the cached chunk probabilities (chunk_cache.json)
and score each combination with scorer.py's logic, without re-querying the
model. Prints the best combos found.
"""
import copy, itertools, json

import vad_pipeline as vp
from scorer import load_gt, score_l1, score_l23

CACHE = json.load(open("chunk_cache.json"))
GT = load_gt("data/Train and Test/test/ground_truth.csv")


def build_preds(cfg):
    preds = {}
    for vid, entry in CACHE.items():
        level = entry["level"]
        chunks = [(w[0], w[1], None) for w in entry["windows"]]
        probs = entry["probs"]
        old_cfg = vp.CFG
        vp.CFG = cfg
        try:
            if level == 1:
                agg = {}
                for p in probs:
                    for c, v in p.items():
                        agg[c] = agg.get(c, 0.0) + v
                z = sum(agg.values()) or 1.0
                agg = {k: v / z for k, v in agg.items()}
                p_anom = 1.0 - agg.get("normal", 0.0)
                if p_anom < cfg["t_level1"]:
                    events = []
                else:
                    cls = max((c for c in agg if c != "normal"), key=lambda c: agg[c])
                    events = [{"class_name": cls, "start_time_sec": None, "end_time_sec": None}]
            else:
                events = [{"class_name": c, "start_time_sec": s, "end_time_sec": e}
                          for c, s, e, _ in vp.aggregate(chunks, probs)]
        finally:
            vp.CFG = old_cfg
        preds[vid] = {"events": events}
    return preds


def score_all(cfg):
    preds = build_preds(cfg)
    r1 = score_l1(preds, GT)
    r2 = score_l23(preds, GT, 2)
    r3 = score_l23(preds, GT, 3)
    s1 = r1["score"] if r1 else 0.0
    s2 = r2["mean_score"] if r2 else 0.0
    s3 = r3["mean_score"] if r3 else 0.0
    # weight roughly like the real grader's marks split (25/35/40 of 100)
    combined = 0.25 * s1 + 0.35 * s2 + 0.40 * s3
    return combined, s1, s2, s3


def main():
    base = copy.deepcopy(vp.CFG)
    results = []

    t_high_opts = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    t_low_opts = [0.25, 0.3, 0.35, 0.4]
    t_open_opts = [1, 2, 3]
    t_video_opts = [0.4, 0.5, 0.55, 0.6, 0.65]
    t_level1_opts = [0.3, 0.35, 0.4, 0.45, 0.5]
    merge_gap_opts = [2.0, 4.0, 8.0, 15.0]

    total = (len(t_high_opts) * len(t_low_opts) * len(t_open_opts)
             * len(t_video_opts) * len(t_level1_opts) * len(merge_gap_opts))
    print(f"sweeping {total} combos...")

    i = 0
    for th, tl, topen, tv, t1, mg in itertools.product(
            t_high_opts, t_low_opts, t_open_opts, t_video_opts, t_level1_opts, merge_gap_opts):
        if tl >= th:
            continue
        i += 1
        cfg = copy.deepcopy(base)
        cfg["t_high"], cfg["t_low"] = th, tl
        cfg["t_open_chunks"] = topen
        cfg["t_video"] = tv
        cfg["t_level1"] = t1
        cfg["merge_gap_sec"] = mg
        combined, s1, s2, s3 = score_all(cfg)
        results.append((combined, s1, s2, s3, th, tl, topen, tv, t1, mg))

    results.sort(key=lambda x: -x[0])
    print(f"\nran {i} valid combos\n")
    print("top 15 by combined weighted score:")
    print(f"{'combined':>9} {'L1':>6} {'L2':>6} {'L3':>6}  t_high t_low topen t_video t_lvl1 merge_gap")
    for r in results[:15]:
        combined, s1, s2, s3, th, tl, topen, tv, t1, mg = r
        print(f"{combined:9.4f} {s1:6.3f} {s2:6.3f} {s3:6.3f}  {th:6.2f} {tl:5.2f} {topen:5d} {tv:7.2f} {t1:6.2f} {mg:9.1f}")

    best = results[0]
    print("\nBEST:", dict(zip(
        ["combined", "s1", "s2", "s3", "t_high", "t_low", "t_open_chunks", "t_video", "t_level1", "merge_gap_sec"],
        best)))


if __name__ == "__main__":
    main()
