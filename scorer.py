"""
Local approximation of the arena's scoring rules, for threshold tuning against
our own held-out labelled set (data/Train and Test/test/ground_truth.csv).

Not a guaranteed match to the hidden grader, but implements the stated rules:
  L1: pooled across all L1 videos, half anomaly-vs-normal accuracy, half class accuracy.
  L2/L3: per-video. Normal GT + no predictions = 1.0. Normal GT + any prediction = 0.0.
         Anomalous GT: did-you-alert + matched-events (IoU>=0.5, each GT event matches
         at most one prediction) + timing quality, averaged into one score per video,
         weighted more toward timing at L3.

Usage:
  python scorer.py --submission sub.json --gt "data/Train and Test/test/ground_truth.csv"
"""
import argparse, csv, json
from collections import defaultdict


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def load_gt(path):
    gt = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            gt[vid].append({
                "level": int(row["level"]),
                "is_anomaly": row["is_anomaly"].strip().lower() == "true",
                "class_name": row["class_name"],
                "start": float(row["start_time_sec"]) if row["start_time_sec"] else None,
                "end": float(row["end_time_sec"]) if row["end_time_sec"] else None,
            })
    return gt


def score_l1(preds, gt):
    anom_correct, cls_correct, n = 0, 0, 0
    for vid, rows in gt.items():
        if rows[0]["level"] != 1:
            continue
        n += 1
        gt_anom = rows[0]["is_anomaly"]
        gt_cls = rows[0]["class_name"]
        p = preds.get(vid, {"events": []})
        pred_events = p.get("events", [])
        pred_anom = len(pred_events) > 0
        if pred_anom == gt_anom:
            anom_correct += 1
        pred_cls = pred_events[0]["class_name"] if pred_events else "normal"
        if pred_cls == gt_cls:
            cls_correct += 1
    if n == 0:
        return None
    return {"n": n, "anomaly_acc": anom_correct / n, "class_acc": cls_correct / n,
            "score": 0.5 * anom_correct / n + 0.5 * cls_correct / n}


def score_video_l23(rows, pred_events, level):
    gt_events = [r for r in rows if r["is_anomaly"]]
    if not gt_events:
        return 1.0 if not pred_events else 0.0

    unmatched_pred = list(range(len(pred_events)))
    matches = []  # (gt_idx, pred_idx, iou)
    for gi, g in enumerate(gt_events):
        best_pi, best_iou = None, 0.0
        for pi in unmatched_pred:
            pe = pred_events[pi]
            if pe["class_name"] != g["class_name"]:
                continue
            if pe.get("start_time_sec") is None or pe.get("end_time_sec") is None:
                continue
            ov = iou(g["start"], g["end"], pe["start_time_sec"], pe["end_time_sec"])
            if ov >= 0.5 and ov > best_iou:
                best_pi, best_iou = pi, ov
        if best_pi is not None:
            matches.append((gi, best_pi, best_iou))
            unmatched_pred.remove(best_pi)

    alerted = 1.0 if pred_events else 0.0
    match_frac = len(matches) / len(gt_events)
    timing = sum(m[2] for m in matches) / len(matches) if matches else 0.0
    false_frags_penalty = min(len(unmatched_pred) * 0.1, 0.3)

    if level == 3:
        w_alert, w_match, w_time = 0.2, 0.4, 0.4
    else:
        w_alert, w_match, w_time = 0.3, 0.45, 0.25
    raw = w_alert * alerted + w_match * match_frac + w_time * timing
    return max(0.0, raw - false_frags_penalty)


def score_l23(preds, gt, level):
    scores = []
    for vid, rows in gt.items():
        if rows[0]["level"] != level:
            continue
        p = preds.get(vid, {"events": []})
        scores.append(score_video_l23(rows, p.get("events", []), level))
    if not scores:
        return None
    return {"n": len(scores), "mean_score": sum(scores) / len(scores),
            "per_video": dict(zip([v for v, r in gt.items() if r[0]["level"] == level], scores))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()

    sub = json.load(open(a.submission))
    preds = {p["video_id"]: p for p in sub["predictions"]}
    gt = load_gt(a.gt)

    r1 = score_l1(preds, gt)
    r2 = score_l23(preds, gt, 2)
    r3 = score_l23(preds, gt, 3)

    print("=== Level 1 ===")
    print(r1)
    print("\n=== Level 2 ===")
    if r2:
        print({"n": r2["n"], "mean_score": round(r2["mean_score"], 3)})
        for vid, s in r2["per_video"].items():
            print(f"  {vid}: {s:.3f}")
    print("\n=== Level 3 ===")
    if r3:
        print({"n": r3["n"], "mean_score": round(r3["mean_score"], 3)})
        for vid, s in r3["per_video"].items():
            print(f"  {vid}: {s:.3f}")


if __name__ == "__main__":
    main()
