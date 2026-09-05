"""
Shot-boundary detection, and whether the cuts coincide with event boundaries.

The level-2/3 videos look like concatenations of source clips - T026 runs through
four unrelated scenes, T025's ground truth is perfectly periodic at 20-40, 60-80,
100-120... If the cut points ARE the event boundaries then segmenting on cuts
gives exact boundaries, which is the whole difficulty at IoU>=0.5: the model can
say *what* confidently but its probabilities are flat across a whole video, so it
cannot say *when*.

Sequential decode, no seeking - per-frame POS_MSEC seeks take minutes on a 10
minute video.

  python shot_cuts.py            # check cuts against known boundaries
"""
import cv2
import numpy as np


def detect(path, thresh=0.35, every=0.25):
    """Return (cut_times, duration). A cut is a big HSV-histogram jump."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps else 0.0
    stride = max(1, int(round(fps * every)))

    prev, cuts, i = None, [], 0
    while True:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, f = cap.retrieve()
            if ok:
                f = cv2.resize(f, (160, 90))
                h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)],
                                 [0, 1], None, [32, 32], [0, 180, 0, 256])
                h = cv2.normalize(h, h).flatten()
                if prev is not None and 1.0 - cv2.compareHist(prev, h, cv2.HISTCMP_CORREL) > thresh:
                    cuts.append(i / fps)
                prev = h
        i += 1
    cap.release()
    return cuts, dur


GT = {
    "T028": [(30, 35), (90, 95), (150, 155), (210, 215)],
    "T025": [(20, 40), (60, 80), (100, 120), (140, 160), (180, 200), (220, 240)],
    "T026": [(10, 40.367), (65, 74.5), (105, 125.8), (150, 210)],
    "T027": [(40, 45), (55, 60), (65, 125), (145, 150)],
    "T033": [(170, 245), (490, 535)],
    "T031": [(235, 360)],
}

if __name__ == "__main__":
    for vid, gt in GT.items():
        cuts, dur = detect(f"data/Train and Test/test/videos/{vid}.mp4")
        bounds = sorted({x for a, b in gt for x in (a, b)})
        print(f"== {vid}  dur {dur:.0f}s  {len(cuts)} cuts", flush=True)
        print(f"   cuts     : {[round(x, 1) for x in cuts][:20]}")
        print(f"   GT bounds: {bounds}")
        if cuts:
            err = [min(abs(x - b) for x in cuts) for b in bounds]
            print(f"   |nearest cut - GT bound|: {[round(e, 1) for e in err]}"
                  f"   median {np.median(err):.2f}s")
