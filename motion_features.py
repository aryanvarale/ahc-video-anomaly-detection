"""
Model-free motion features, sampled on a fixed time grid per video.

The fine-tuned VLM tells us *what* a video contains but, for scene-property
anomalies like congestion, its per-chunk probability is flat 1.000 across every
chunk - it was trained on trimmed clips where the whole clip is the anomaly, so
it answers "what kind of scene is this" rather than "is it happening now".
That leaves no signal to segment on.

Motion carries the missing *when*. Congestion is vehicles that stop moving;
a collision is a sharp motion discontinuity. Both are visible in optical flow
without any model at all.

Features per grid step (default every 0.5 s):
  mean, p90       - dense-flow magnitude, raw
  res_mean, res_p90 - the same after subtracting the frame's median flow vector,
                    which cancels a panning/dashcam camera and leaves only motion
                    relative to the scene
  moving_frac     - fraction of pixels whose residual flow exceeds 0.5 px
  diff            - mean absolute frame difference, a cheap texture-change proxy

  python motion_features.py --videos "data/Train and Test/test/videos" --out motion.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def video_features(path, step=0.5, width=320):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = nframes / fps if fps > 0 else 0.0

    times, feats = [], []
    prev = None
    t = 0.0
    while duration <= 0 or t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2)
            # Median flow vector ~ camera egomotion; what is left is scene motion.
            med = np.median(flow.reshape(-1, 2), axis=0)
            res = np.linalg.norm(flow - med, axis=2)
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32))))
            feats.append({
                "mean": float(mag.mean()), "p90": float(np.percentile(mag, 90)),
                "res_mean": float(res.mean()), "res_p90": float(np.percentile(res, 90)),
                "moving_frac": float((res > 0.5).mean()), "diff": diff,
            })
            times.append(t)
        prev = gray
        t += step

    cap.release()
    return {"fps": fps, "duration": duration, "step": step,
            "times": times, "feats": feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()

    vids = sorted(Path(a.videos).glob("*.mp4"))
    if a.only:
        vids = [v for v in vids if v.stem in a.only]

    out = {}
    outp = Path(a.out)
    if outp.exists():
        out = json.load(open(outp))

    for v in vids:
        if v.stem in out:
            print(f"{v.stem}: cached", flush=True)
            continue
        t0 = time.time()
        out[v.stem] = video_features(v, a.step)
        json.dump(out, open(outp, "w"))
        print(f"{v.stem}: {len(out[v.stem]['times'])} steps, "
              f"{out[v.stem]['duration']:.0f}s video, {time.time()-t0:.1f}s", flush=True)

    print(f"wrote {outp} ({len(out)} videos)")


if __name__ == "__main__":
    main()
