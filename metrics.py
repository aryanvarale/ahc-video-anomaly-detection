"""
Event-level precision / recall / F1 against our held-out ground truth.

Mirrors how the arena reports detection quality, so our local numbers are
comparable to the ones it gives back: an event counts as found only when the
class matches AND (at Levels 2-3) temporal IoU >= 0.5. Level-1 rows carry no
timestamps, so there it is class match alone.

  python metrics.py --submission submission_ft_v2.json
"""
import argparse
import csv
from collections import defaultdict

IOU_GATE = 0.5


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def load_gt(path):
    gt = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            gt[r["video_id"]].append({
                "level": int(r["level"]),
                "is_anomaly": r["is_anomaly"].strip().lower() == "true",
                "cls": r["class_name"],
                "start": float(r["start_time_sec"]) if r["start_time_sec"] else None,
                "end": float(r["end_time_sec"]) if r["end_time_sec"] else None,
            })
    return gt


def score(preds, gt):
    tp = fp = fn = 0
    per_class = defaultdict(lambda: {"found": 0, "total": 0, "false": 0})
    fp_videos = []

    for vid, rows in gt.items():
        level = rows[0]["level"]
        truths = [r for r in rows if r["is_anomaly"]]
        guesses = list(preds.get(vid, {}).get("events", []))

        for t in truths:
            per_class[t["cls"]]["total"] += 1

        unmatched = list(range(len(guesses)))
        for t in truths:
            hit = None
            for gi in unmatched:
                g = guesses[gi]
                if g["class_name"] != t["cls"]:
                    continue
                if level == 1:
                    hit = gi
                    break
                if (g.get("start_time_sec") is not None
                        and iou(t["start"], t["end"],
                                g["start_time_sec"], g["end_time_sec"]) >= IOU_GATE):
                    hit = gi
                    break
            if hit is None:
                fn += 1
            else:
                tp += 1
                per_class[t["cls"]]["found"] += 1
                unmatched.remove(hit)

        for gi in unmatched:                      # predictions nothing matched
            fp += 1
            per_class[guesses[gi]["class_name"]]["false"] += 1
            fp_videos.append((vid, guesses[gi]["class_name"]))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "per_class": per_class, "fp_videos": fp_videos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission_ft_v2.json")
    ap.add_argument("--gt", default="data/Train and Test/test/ground_truth.csv")
    a = ap.parse_args()

    import json
    preds = {p["video_id"]: p for p in json.load(open(a.submission))["predictions"]}
    r = score(preds, load_gt(a.gt))

    print(f"=== {a.submission} ===\n")
    print(f"  precision   {r['precision']*100:5.1f}%   (of what we flagged, this much was real)")
    print(f"  recall      {r['recall']*100:5.1f}%   (of what happened, we found this much)")
    print(f"  F1          {r['f1']*100:5.1f}%")
    print(f"  TP {r['tp']}   FP {r['fp']} (false alarms)   FN {r['fn']} (missed)\n")

    print("  per class:")
    rows = sorted(r["per_class"].items(), key=lambda kv: (-kv[1]["total"], kv[0]))
    for cls, d in rows:
        if not d["total"] and not d["false"]:
            continue
        rate = f"{d['found']/d['total']*100:3.0f}%" if d["total"] else "  —"
        print(f"    {cls:36s} found {d['found']}/{d['total']}   {rate}   false {d['false']}")

    if r["fp_videos"]:
        print("\n  false alarms by video:")
        for vid, cls in r["fp_videos"]:
            print(f"    {vid}  {cls}")


if __name__ == "__main__":
    main()
