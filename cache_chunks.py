"""
Run every video through read_chunks + classify ONCE and cache the raw
per-chunk probability vectors + timing to disk, so threshold sweeps can
replay aggregate() offline without re-querying the model each time.

Usage:
  python cache_chunks.py --manifest manifest.json --videos "data/Train and Test/test/videos" --out chunk_cache.json
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor

import vad_pipeline as vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="chunk_cache.json")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    rows = man["videos"] if isinstance(man, dict) and "videos" in man else man
    levels = {r["video_id"]: int(r.get("level", 1)) for r in rows}
    files = {r["video_id"]: r.get("filename", r.get("file", r["video_id"] + ".mp4"))
             for r in rows}

    cache = {}
    with ThreadPoolExecutor(vp.CFG["concurrency"]) as pool:
        for i, vid in enumerate(sorted(levels), 1):
            fname = os.path.basename(files[vid])
            path = os.path.join(a.videos, fname)
            if not os.path.exists(path):
                print(f"!! missing {path}")
                continue
            level = levels[vid]
            chunks, dur, nframes = vp.read_chunks(path, level)
            probs = list(pool.map(lambda c: vp.classify(c[2]), chunks))
            cache[vid] = {
                "level": level,
                "duration": dur,
                "nframes": nframes,
                "windows": [[c[0], c[1]] for c in chunks],
                "probs": probs,
            }
            print(f"[{i}/{len(levels)}] {vid} L{level} -> {len(chunks)} chunks cached")

    json.dump(cache, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
