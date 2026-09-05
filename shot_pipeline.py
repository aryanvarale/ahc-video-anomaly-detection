"""
Shot-segmented detection for levels 2 and 3.

Measured against ground truth, shot cuts ARE the event boundaries on the videos
that are built by concatenating source clips:

    T028   8 cuts -> [30.0, 35.0, 90.0, 95.0, 150.0, 155.0, 210.0, 215.0]
           8 GT   -> [30,   35,   90,   95,   150,   155,   210,   215  ]   err 0.02s
    T026   8 cuts -> [10.1, 40.6, 65.0, 74.6, 105.1, 126.0, 150.2, 210.2]
           8 GT   -> [10,   40.4, 65,   74.5, 105,   125.8, 150,   210  ]   err 0.17s

So the segmentation problem is already solved by a histogram diff, for free.
What is NOT solved is the label: on T026 the sliding-window pipeline got all four
boundaries essentially right and three of the four classes wrong, which is why it
scored 0.343.

Hence this classifies a whole shot as one unit - averaging the model over every
window in the shot instead of letting a 2s window open an event - and emits the
shot's own boundaries. It is also cheaper: T026 costs ~70 calls here against 240
for the sliding window.

Videos with no cuts (T027, T031: congestion developing inside one continuous
shot) fall back to the existing hysteresis path, so nothing regresses there.
"""
import base64

import cv2

import vad_pipeline as vp
from shot_cuts import detect


def segments(path, min_len=2.0, thresh=0.35):
    """Cut list -> [(start, end)] covering the whole video.

    Shots shorter than min_len are folded into the previous one: a 0.5s shot has
    too few frames to classify, and emitting it as an event would just add an
    unmatched prediction, which the scorer charges 0.1 for."""
    cuts, dur = detect(path, thresh=thresh)
    if dur <= 0:
        return [], 0.0
    bounds = [0.0] + [c for c in cuts if 0.0 < c < dur] + [dur]
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        if segs and b - a < min_len:
            segs[-1] = (segs[-1][0], b)
        elif b - a > 0.01:
            segs.append((a, b))
    return segs, dur


def _frames_for(path, spans, k):
    """Encode k frames for each (start, end) span in ONE sequential decode."""
    fps, nframes, dur = vp.probe(path)
    need = {}
    for si, (a, b) in enumerate(spans):
        for j in range(k):
            idx = int(round((a + (b - a) * (j + 0.5) / k) * fps))
            need.setdefault(min(max(idx, 0), max(nframes - 1, 0)), []).append(si)

    got, cap, i, ptr = {}, cv2.VideoCapture(path), 0, 0
    target = sorted(need)
    while ptr < len(target):
        if not cap.grab():
            break
        if i == target[ptr]:
            ok, frame = cap.retrieve()
            if ok:
                got[i] = vp.encode(frame, vp.CFG["long_side"])
            ptr += 1
            while ptr < len(target) and target[ptr] == i:
                ptr += 1
        i += 1
    cap.release()

    out = [[] for _ in spans]
    for idx in sorted(got):
        if got[idx]:
            for si in need.get(idx, []):
                out[si].append(got[idx])
    return out


def classify_shots(path, segs, pool, windows_per_shot=6, k=8):
    """Average the model over several windows spread across each shot."""
    spans, owner = [], []
    for si, (a, b) in enumerate(segs):
        n = max(1, min(windows_per_shot, int((b - a) // 2) or 1))
        for w in range(n):
            s = a + (b - a) * w / n
            e = min(s + 2.0, b)
            if e - s < 0.4:
                continue
            spans.append((s, e))
            owner.append(si)

    imgs = _frames_for(path, spans, k)
    live = [(sp, im, ow) for sp, im, ow in zip(spans, imgs, owner) if im]
    timed = list(pool.map(lambda t: vp._timed_classify(t[1]), live))

    acc = [{} for _ in segs]
    cnt = [0] * len(segs)
    for (_, _, si), (probs, _) in zip(live, timed):
        for c, v in probs.items():
            acc[si][c] = acc[si].get(c, 0.0) + v
        cnt[si] += 1

    out = []
    for si in range(len(segs)):
        if not cnt[si]:
            out.append(None)
            continue
        z = sum(acc[si].values()) or 1.0
        out.append({c: v / z for c, v in acc[si].items()})
    return out, [t for _, t in timed]
