"""
Split an SFT jsonl into train/val *by source video*.

Splitting by sample would leak: chunks cut from one video are near-duplicates of
each other, so the same video landing on both sides makes val accuracy measure
memorisation rather than generalisation. The frame paths are named
'<video_basename>_c<chunk>_<frame>.jpg', so the basename recovers the video.

  python split_sft.py --in sft_bal.jsonl --train sft_bal_train.jsonl \
                      --val sft_bal_val.jsonl --val_frac 0.08
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict


def source_of(rec):
    p = os.path.basename(rec["images"][0])
    m = re.match(r"(.+)_c\d{4}_\d+\.jpg$", p)
    return m.group(1) if m else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp)]

    # A video contributes chunks under exactly one label, so grouping by
    # (label, source) and holding out whole sources keeps the split clean.
    by_label = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label[r["messages"][-1]["content"].strip()][source_of(r)].append(r)

    rnd = random.Random(a.seed)
    train, val = [], []
    for lab, srcs in sorted(by_label.items()):
        names = sorted(srcs)
        rnd.shuffle(names)
        n_val = max(1, round(len(names) * a.val_frac))
        for nm in names[:n_val]:
            val += srcs[nm]
        for nm in names[n_val:]:
            train += srcs[nm]
        print(f"  {lab}: {len(names)} videos -> {n_val} val")

    rnd.shuffle(train)
    rnd.shuffle(val)
    for path, data in ((a.train, train), (a.val, val)):
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    print(f"\ntrain {len(train)}  {dict(sorted(Counter(r['messages'][-1]['content'] for r in train).items()))}")
    print(f"val   {len(val)}  {dict(sorted(Counter(r['messages'][-1]['content'] for r in val).items()))}")


if __name__ == "__main__":
    main()
