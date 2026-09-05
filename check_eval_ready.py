"""
Pre-flight check before running the eval pack: are the videos actually there,
and does what is on disk match the manifest?

Worth running first because a missing file is silent in the worst way - the
pipeline skips it, the arena scores an unanswered video as `normal`, and an
anomalous video lost that way costs the full mark without any error appearing.

  python check_eval_ready.py --manifest manifest_eval.json --videos ./eval_videos
"""
import argparse
import json
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_eval.json")
    ap.add_argument("--videos", required=True)
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    rows = man["videos"] if isinstance(man, dict) and "videos" in man else man

    missing, ok, mismatch = [], [], []
    total_declared = 0.0
    for r in rows:
        vid = r["video_id"]
        name = r.get("filename", r.get("file", vid + ".mp4"))
        path = os.path.join(a.videos, name)
        declared = float(r.get("duration_sec") or 0)
        total_declared += declared
        if not os.path.exists(path):
            missing.append((vid, name))
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        actual = n / fps if fps else 0.0
        ok.append(vid)
        # a real mismatch means we would be timestamping against the wrong clip
        if declared and abs(actual - declared) > max(1.0, 0.05 * declared):
            mismatch.append((vid, declared, round(actual, 1)))

    print(f"manifest: {len(rows)} videos, {total_declared/60:.1f} min declared")
    print(f"present : {len(ok)}")
    if missing:
        print(f"MISSING : {len(missing)}  -> these score as `normal` if not answered")
        for vid, name in missing[:40]:
            print(f"   {vid}  (looked for {name})")
    if mismatch:
        print(f"DURATION MISMATCH: {len(mismatch)} (declared vs actual)")
        for vid, d, act in mismatch[:20]:
            print(f"   {vid}: manifest {d}s, file {act}s")
    if not missing and not mismatch:
        by_level = {}
        for r in rows:
            by_level[int(r.get("level", 1))] = by_level.get(int(r.get("level", 1)), 0) + 1
        print(f"ready. levels: {dict(sorted(by_level.items()))}")
        # 2s windows at L1/L2, 3s at L3, minus the stride overlap
        est = sum(float(r.get("duration_sec") or 0) / (1.0 if int(r.get("level", 1)) < 3 else 1.5)
                  for r in rows)
        print(f"~{est:.0f} model calls; at ~0.45 s/call over 8 workers, "
              f"roughly {est*0.45/8/60:.1f} min of inference")


if __name__ == "__main__":
    main()
