"""
Per-class accuracy and confusion matrix on a held-out SFT split.

The arena only reports three aggregate numbers, which is far too coarse to tell
whether a change fixed the thing it was aimed at. This runs the served model over
sft_*_val.jsonl - videos no training step ever saw - and prints the full
confusion matrix, so a claim like "balancing stops fighting being swallowed by
loitering" is checked directly rather than inferred from a score moving.

  VAD_SERVER=http://127.0.0.1:28451/v1/chat/completions python eval_val.py \
      --val sft_bal_val.jsonl --limit 1200
"""
import argparse
import base64
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import vad_pipeline as vp


def predict(rec):
    imgs = []
    for p in rec["images"]:
        with open(p, "rb") as f:
            imgs.append(base64.b64encode(f.read()).decode())
    # classify() applies temperature scaling, which is monotonic and so cannot
    # change the argmax - fine for accuracy, and it keeps this identical to the
    # path the submission actually uses.
    probs = vp.classify(imgs)
    return max(probs, key=probs.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--limit", type=int, default=0, help="sample this many (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.val)]
    if a.limit and len(rows) > a.limit:
        # stratified: an unbalanced subsample would misreport the rare classes,
        # which are exactly the ones under investigation
        by = defaultdict(list)
        for r in rows:
            by[r["messages"][-1]["content"].strip()].append(r)
        per = max(1, a.limit // len(by))
        rnd = random.Random(0)
        rows = []
        for lab, v in by.items():
            rnd.shuffle(v)
            rows += v[:per]

    gold = [r["messages"][-1]["content"].strip() for r in rows]
    with ThreadPoolExecutor(a.workers) as ex:
        pred_classes = list(ex.map(predict, rows))
    c2l = {v: k for k, v in vp.L2C.items()}
    pred = [c2l.get(c, "?") for c in pred_classes]

    labels = sorted(set(gold) | set(p for p in pred if p != "?"))
    conf = Counter(zip(gold, pred))
    correct = sum(v for (g, p), v in conf.items() if g == p)

    print(f"\noverall: {correct}/{len(gold)} = {correct/len(gold)*100:.1f}%\n")
    print("rows = truth, cols = prediction")
    hdr = "      " + "".join(f"{l:>5}" for l in labels) + "   acc   n   class"
    print(hdr)
    per_class = {}
    for g in labels:
        n = sum(v for (gg, _), v in conf.items() if gg == g)
        c = conf.get((g, g), 0)
        acc = c / n if n else 0.0
        per_class[g] = acc
        cells = "".join(f"{conf.get((g,p),0):>5}" for p in labels)
        print(f"  {g}   {cells}  {acc*100:5.1f} {n:>4}   {vp.L2C.get(g,'?')}")

    worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
    print("\nweakest: " + ", ".join(f"{vp.L2C.get(k,k)} {v*100:.0f}%" for k, v in worst))

    if a.out:
        json.dump({"overall": correct / len(gold),
                   "per_class": {vp.L2C.get(k, k): v for k, v in per_class.items()},
                   "confusion": {f"{g}->{p}": v for (g, p), v in conf.items()}},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
