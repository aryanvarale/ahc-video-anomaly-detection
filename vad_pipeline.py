"""
AHC Video Anomaly Detection - inference + submission pipeline.

Design:
  Qwen3-VL-4B-Instruct (LoRA fine-tuned, merged) served by vLLM.
  Each video is cut into overlapping chunks of N frames. The model answers with
  ONE letter (A-L). We read the top_logprobs of that single token to get a
  calibrated probability over the 12 classes -> thresholding + hysteresis.

Usage:
  # 1. start server (see README section in the plan)
  # 2. python vad_pipeline.py --manifest manifest.json --videos ./videos \
  #        --template starter_template.json --out submission.json
"""

import argparse, base64, json, math, os, statistics, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

# --------------------------------------------------------------------------- #
# config - THIS IS WHAT YOU TUNE. Everything else can stay as is.
# --------------------------------------------------------------------------- #
CFG = {
    "server": os.environ.get("VAD_SERVER", "http://127.0.0.1:28451/v1/chat/completions"),
    "model": "vad-qwen3vl-4b",

    # chunking
    # Matching L3 to the 2.0s geometry the LoRA trains on looked like a clear win
    # (L3 0.079 -> 0.189) but that baseline was a STALE CACHE. Re-measured against
    # the shipped v4 configuration it is a regression - 3.0s scores 0.216, 2.0s
    # scores 0.189 - so L3 stays at 3.0s. Worth retrying against a retrained model,
    # but only with the baseline re-measured, not read off an old cache.
    # Level 1 chunks densely (2s/1s) rather than taking one whole-clip pass. A
    # single pass has to compress a 26s clip into 16 frames, and a brief anomaly
    # gets averaged away; 2s windows also match the geometry the LoRA trained on.
    # Measured on the held-out set, this took level 1 from 5/20 to 12/20 correct,
    # and it is the configuration the shipped submission was built with - so it
    # is the default here, or running this file would not reproduce that score.
    "chunk_sec": {1: 2.0, 2: 2.0, 3: 3.0},
    "stride_sec": {1: 1.0, 2: 1.0, 3: 1.5},
    "frames_per_chunk": {1: 8, 2: 8, 3: 8},
    "long_side": 448,                             # frame resize, keeps tokens low

    # decision thresholds (tune on your val split - biggest score lever)
    # Training to ~100% token accuracy leaves the model overconfident: its
    # per-chunk probabilities saturate at 0/1, so long stretches read as one
    # undifferentiated block and event boundaries can't be cut from it (measured:
    # with T=1 every threshold setting scored identically - thresholds were inert).
    # Softening the distribution restores the dynamic range they act on.
    "temperature": 2.5,      # 1.0 = off. Chosen mid-plateau (stable over 2.0-3.0)
                             # rather than at the sharper single-point optimum.
    "t_high": 0.65,          # open an event
    "t_low": 0.30,           # close an event
    "t_open_chunks": 2,      # consecutive chunks above t_high needed to open
    "t_video": 0.65,         # video-level gate: below this -> events: []
    # L1: p(anomaly) needed to leave "normal".
    # Swept on the held-out level-1 videos: 0.15-0.40 all score 0.708, 0.45-0.50
    # score 0.750, 0.60 drops to 0.729. The step at 0.45 is T003, a normal video
    # sitting at p_anom 0.42 - below 0.45 it becomes a false alarm, which at level
    # 1 costs BOTH halves of the score (anomaly-vs-normal and class). 0.50 is the
    # middle of the good plateau rather than its edge.
    "t_level1": 0.50,
    "merge_gap_sec": 2.0,    # same-class runs closer than this are merged
    "max_events": 4,         # fragmentation guard

    # --- stage 2: tiled verification (off by default) ------------------------
    # The always-on pass sees the whole frame downscaled to `long_side`, so an
    # anomaly occupying a small corner of a wide aerial shot can collapse into a
    # couple of patches and vanish. This second stage re-reads those frames at
    # full resolution, splits them into a grid, and classifies each tile, giving
    # a small region grid^2 times the pixel density it had in the full frame.
    # It only fires on a minority of chunks, so the real-time budget holds.
    "verify_enabled": False,
    "verify_band": (0.20, 0.80),   # first-pass p(anomaly) in this band -> uncertain, verify
    "verify_every_n": 6,           # ALSO verify every Nth chunk regardless of confidence:
                                   # a small anomaly the low-res pass confidently calls
                                   # "normal" never lands in the uncertain band, and the
                                   # classes most likely to be small (debris, loitering,
                                   # stalled vehicle) are also the most persistent, so
                                   # periodic sampling has a good chance of catching them.
    "verify_grid": 2,              # 2x2 tiling
    "verify_max_calls": 60,        # hard per-video cap on extra model calls
    "verify_tile_min_conf": 0.90,  # tiles are out-of-distribution vs the full-frame
                                   # training data, so demand high confidence before a
                                   # tile is allowed to override the full-frame verdict
    "min_dur_sec": {         # per-class minimum plausible duration
        "traffic_accident": 1.0,
        "traffic_congestion": 4.0,
        "stalled_or_broken_down_vehicle": 6.0,
        "vehicle_blocking_traffic": 4.0,
        "fire": 2.0,
        "smoke": 2.0,
        "waterlogging_or_flood": 3.0,
        "wrong_way_driving": 2.0,
        "road_spill_or_debris": 2.0,
        "fighting_or_violence": 2.0,
        "loitering_or_suspicious_presence": 5.0,
    },
    "concurrency": 16,       # parallel requests to vLLM
}

LETTERS = "ABCDEFGHIJKL"
CLASSES = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "fire",
    "smoke",
    "waterlogging_or_flood",
    "wrong_way_driving",
    "road_spill_or_debris",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
L2C = dict(zip(LETTERS, CLASSES))

PROMPT = (
    "You are watching a short sequence of consecutive frames from an aerial drone, "
    "CCTV or dashcam feed over an urban area. Decide what is happening in THIS clip.\n"
    "A normal - ordinary moving traffic, parked cars in bays, pedestrians going about, empty road\n"
    "B traffic_accident - collision, crash, overturned vehicle, impact aftermath\n"
    "C traffic_congestion - lanes densely packed, vehicles barely moving, queue building\n"
    "D stalled_or_broken_down_vehicle - vehicle stationary on a carriageway or shoulder where it should not be\n"
    "E vehicle_blocking_traffic - vehicle obstructing a lane, junction or gate and holding others up\n"
    "F fire - visible flames\n"
    "G smoke - smoke without clear flames\n"
    "H waterlogging_or_flood - standing water covering the road\n"
    "I wrong_way_driving - a vehicle travelling against the direction of traffic\n"
    "J road_spill_or_debris - spilled load, rubble, fallen object or debris on the road\n"
    "K fighting_or_violence - people physically fighting or assaulting\n"
    "L loitering_or_suspicious_presence - person lingering where nobody should be, climbing, tampering\n\n"
    "Be strict: if it is ordinary activity, answer A. Answer with exactly one letter."
)

# --------------------------------------------------------------------------- #
# frame extraction
# --------------------------------------------------------------------------- #
def probe(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n, (n / fps if fps else 0.0)


def encode(frame, long_side):
    h, w = frame.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        frame = cv2.resize(frame, (int(w * s) // 28 * 28, int(h * s) // 28 * 28))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


def read_chunks(path, level):
    """Yield (start_sec, end_sec, [b64 jpeg, ...])."""
    fps, nframes, dur = probe(path)
    chunk_sec, stride = CFG["chunk_sec"][level], CFG["stride_sec"][level]
    k = CFG["frames_per_chunk"][level]

    if chunk_sec is None:                      # Level 1: one pass over the clip
        windows = [(0.0, max(dur, 0.1))]
    else:
        windows, t = [], 0.0
        while t < max(dur - 0.5, 0.1):
            windows.append((t, min(t + chunk_sec, dur)))
            t += stride
        if not windows:
            windows = [(0.0, max(dur, 0.1))]

    # single sequential decode pass, collecting every frame index we need
    need = {}
    for wi, (a, b) in enumerate(windows):
        for j in range(k):
            idx = int(round((a + (b - a) * (j + 0.5) / k) * fps))
            need.setdefault(min(max(idx, 0), max(nframes - 1, 0)), []).append(wi)

    got, cap, i = {}, cv2.VideoCapture(path), 0
    target = sorted(need)
    ptr = 0
    while ptr < len(target):
        ok = cap.grab()
        if not ok:
            break
        if i == target[ptr]:
            ok, frame = cap.retrieve()
            if ok:
                got[i] = encode(frame, CFG["long_side"])
            ptr += 1
            while ptr < len(target) and target[ptr] == i:
                ptr += 1
        i += 1
    cap.release()

    out, used = [], 0
    for wi, (a, b) in enumerate(windows):
        imgs = [got[idx] for idx in sorted(got) if wi in need.get(idx, [])]
        imgs = [x for x in imgs if x]
        if imgs:
            out.append((a, b, imgs))
            used += len(imgs)
    return out, dur, used


# --------------------------------------------------------------------------- #
# stage 2: tiled verification
# --------------------------------------------------------------------------- #
def read_frames_raw(path, start_sec, end_sec, k):
    """Re-read this chunk's k frames at FULL resolution.

    read_chunks() throws the raw frames away (it encodes straight to downscaled
    jpeg) and holding them all would cost gigabytes on a long clip, so the
    verification stage pays a second seek+decode for the chunks it actually
    looks at."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for j in range(k):
        t = start_sec + (end_sec - start_sec) * (j + 0.5) / k
        idx = min(max(int(round(t * fps)), 0), max(nframes - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def tile_frame(frame, grid):
    """Split into grid x grid tiles -> [((row, col), tile), ...].
    The last row/column absorbs any remainder so no pixels are dropped."""
    h, w = frame.shape[:2]
    th, tw = h // grid, w // grid
    if th < 1 or tw < 1:
        return [((0, 0), frame)]
    tiles = []
    for r in range(grid):
        for c in range(grid):
            y0, x0 = r * th, c * tw
            y1 = h if r == grid - 1 else (r + 1) * th
            x1 = w if c == grid - 1 else (c + 1) * tw
            tiles.append(((r, c), frame[y0:y1, x0:x1]))
    return tiles


def verify_chunk(path, start_sec, end_sec, k, classify_fn=None):
    """Classify each tile position of this chunk separately.

    Tiles are grouped BY POSITION across the chunk's frames, so every model call
    still sees a short temporal sequence of one region rather than a single
    still - that keeps the input shaped like what the model was trained on.

    Returns (best_probs, call_times_ms). best_probs is the tile distribution
    with the strongest anomaly signal, or None if nothing could be read."""
    classify_fn = classify_fn or classify
    frames = read_frames_raw(path, start_sec, end_sec, k)
    if not frames:
        return None, []

    by_pos = {}
    for fr in frames:
        for pos, tile in tile_frame(fr, CFG["verify_grid"]):
            by_pos.setdefault(pos, []).append(tile)

    best, times = None, []
    for _, tiles in sorted(by_pos.items()):
        imgs = [x for x in (encode(t, CFG["long_side"]) for t in tiles) if x]
        if not imgs:
            continue
        t0 = time.perf_counter()
        probs = classify_fn(imgs)
        times.append((time.perf_counter() - t0) * 1000.0)
        if probs and (best is None or
                      1.0 - probs.get("normal", 0.0) > 1.0 - best.get("normal", 0.0)):
            best = probs
    return best, times


def wants_verify(p_anom, chunk_idx):
    lo, hi = CFG["verify_band"]
    if lo < p_anom < hi:
        return True
    n = CFG["verify_every_n"]
    return bool(n) and chunk_idx % n == 0


def apply_verification(path, chunks, probs, level):
    """Run stage 2 over the chunks that warrant it and fold the results in.

    Returns (probs, extra_call_times_ms). probs is modified in place-ish (a new
    list is returned) - a tile only overrides the full-frame verdict when it is
    both more anomalous AND above verify_tile_min_conf, because a cropped tile is
    out-of-distribution relative to the full-frame chunks the model was tuned on
    and its confidence is correspondingly less trustworthy."""
    k = CFG["frames_per_chunk"][level]
    out = list(probs)
    extra_times, budget = [], CFG["verify_max_calls"]

    for i, (a, b, _) in enumerate(chunks):
        if budget <= 0:
            break
        full_anom = 1.0 - out[i].get("normal", 0.0)
        if not wants_verify(full_anom, i):
            continue
        tile_probs, times = verify_chunk(path, a, b, k)
        extra_times += times
        budget -= max(len(times), 1)
        if not tile_probs:
            continue
        tile_anom = 1.0 - tile_probs.get("normal", 0.0)
        if tile_anom > full_anom and tile_anom >= CFG["verify_tile_min_conf"]:
            out[i] = tile_probs
    return out, extra_times


# --------------------------------------------------------------------------- #
# model call -> probability vector
# --------------------------------------------------------------------------- #
def classify(imgs, retries=2):
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in imgs]
    content.append({"type": "text", "text": PROMPT})
    body = {
        "model": CFG["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(CFG["server"], json=body, timeout=120)
            r.raise_for_status()
            tl = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            raw = {}
            for e in tl:
                t = e["token"].strip().upper()[:1]
                if t in LETTERS:
                    raw[t] = max(raw.get(t, -1e9), e["logprob"])
            if not raw:
                raw = {"A": 0.0}
            m = max(raw.values())
            ex = {k: math.exp(v - m) for k, v in raw.items()}
            z = sum(ex.values())
            return apply_temperature({L2C[k]: v / z for k, v in ex.items()})
        except Exception:
            if attempt == retries:
                return {"normal": 1.0}
            time.sleep(1.0)


# --------------------------------------------------------------------------- #
# temporal aggregation
# --------------------------------------------------------------------------- #
def smooth(xs, w=3):
    if len(xs) < w:
        return xs
    h = w // 2
    return [statistics.median(xs[max(0, i - h): i + h + 1]) for i in range(len(xs))]


def aggregate(chunks, probs):
    """chunks: [(a,b,_)], probs: [dict]  ->  [(class, start, end, score)]"""
    anom = smooth([1.0 - p.get("normal", 0.0) for p in probs])
    if not anom or max(anom) < CFG["t_video"]:
        return []

    runs, i, n = [], 0, len(anom)
    while i < n:
        if anom[i] >= CFG["t_high"]:
            j, ok = i, 0
            while j < n and anom[j] >= CFG["t_high"] and ok < CFG["t_open_chunks"]:
                j += 1; ok += 1
            if ok < CFG["t_open_chunks"] and n >= CFG["t_open_chunks"]:
                i += 1; continue
            while j < n and anom[j] >= CFG["t_low"]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    events = []
    for a, b in runs:
        agg = {}
        for p in probs[a: b + 1]:
            for c, v in p.items():
                if c != "normal":
                    agg[c] = agg.get(c, 0.0) + v
        if not agg:
            continue
        cls = max(agg, key=agg.get)
        s, e = chunks[a][0], chunks[b][1]
        events.append([cls, round(s, 2), round(e, 2),
                       max(anom[a: b + 1])])

    # merge same-class neighbours to avoid fragmentation penalty
    events.sort(key=lambda x: x[1])
    merged = []
    for ev in events:
        if merged and merged[-1][0] == ev[0] and ev[1] - merged[-1][2] <= CFG["merge_gap_sec"]:
            merged[-1][2] = max(merged[-1][2], ev[2])
            merged[-1][3] = max(merged[-1][3], ev[3])
        else:
            merged.append(list(ev))

    out = [e for e in merged
           if e[2] - e[1] >= CFG["min_dur_sec"].get(e[0], 1.0)]
    out.sort(key=lambda x: -x[3])
    return sorted(out[: CFG["max_events"]], key=lambda x: x[1])


EXPL = {
    "traffic_accident": "Vehicles collide and come to an abrupt stop; the impact and immediate aftermath are visible in this segment of the clip.",
    "traffic_congestion": "Lanes are densely packed with vehicles moving very slowly or not at all, with a queue extending back through the frame.",
    "stalled_or_broken_down_vehicle": "A vehicle remains stationary on the carriageway or shoulder while traffic continues past it, consistent with a breakdown.",
    "vehicle_blocking_traffic": "A vehicle is stopped across a lane or junction and is obstructing the movement of other traffic behind it.",
    "fire": "Open flames are visible in the scene, with a bright fire front and associated smoke rising from the source.",
    "smoke": "A plume of smoke rises through the frame without clearly visible flames at the source of the emission.",
    "waterlogging_or_flood": "Standing water covers the road surface, with vehicles moving slowly through it or avoiding the flooded section.",
    "wrong_way_driving": "A vehicle travels against the prevailing direction of traffic, moving head-on relative to the surrounding flow.",
    "road_spill_or_debris": "Material is spilled or scattered across the road surface, obstructing part of the carriageway in this segment.",
    "fighting_or_violence": "People are engaged in a physical altercation, with visible grappling or striking between individuals in the scene.",
    "loitering_or_suspicious_presence": "A person lingers in an area with no ordinary reason to be present, moving around rather than passing through.",
}


# --------------------------------------------------------------------------- #
# per-video driver
# --------------------------------------------------------------------------- #
def apply_temperature(probs, T=None):
    """Soften a saturated class distribution: p^(1/T), renormalised.

    T > 1 flattens. Applied to the already-normalised probabilities rather than
    the raw logits, which is equivalent up to the constant that normalising
    removes anyway."""
    T = CFG["temperature"] if T is None else T
    if not T or T == 1.0:
        return probs
    s = {k: max(v, 1e-12) ** (1.0 / T) for k, v in probs.items()}
    z = sum(s.values()) or 1.0
    return {k: v / z for k, v in s.items()}


def _timed_classify(imgs):
    t = time.perf_counter()
    result = classify(imgs)
    return result, (time.perf_counter() - t) * 1000.0


def _percentile(sorted_xs, p):
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_xs[int(k)]
    return sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


def do_video(vid, path, level, pool):
    t0 = time.perf_counter()
    chunks, dur, nframes = read_chunks(path, level)
    timed = list(pool.map(lambda c: _timed_classify(c[2]), chunks))
    calls = [r for r, _ in timed]
    call_times = [t for _, t in timed]

    if CFG["verify_enabled"] and chunks:
        calls, extra_times = apply_verification(path, chunks, calls, level)
        call_times += extra_times   # verification is real work; it must show up
                                    # in call_count/total_time_ms, since the
                                    # latency bonus is computed from what we report
    call_times.sort()

    if level == 1:
        agg = {}
        for p in calls:
            for c, v in p.items():
                agg[c] = agg.get(c, 0.0) + v
        z = sum(agg.values()) or 1.0
        agg = {k: v / z for k, v in agg.items()}
        p_anom = 1.0 - agg.get("normal", 0.0)
        if p_anom < CFG["t_level1"]:
            events = []
        else:
            cls = max((c for c in agg if c != "normal"), key=lambda c: agg[c])
            events = [{"class_name": cls, "start_time_sec": None,
                       "end_time_sec": None, "explanation": EXPL[cls]}]
    else:
        events = [{"class_name": c, "start_time_sec": s, "end_time_sec": e,
                   "explanation": EXPL[c]} for c, s, e, _ in aggregate(chunks, calls)]

    ms = (time.perf_counter() - t0) * 1000.0
    n = max(len(call_times), 1)
    total_call_ms = sum(call_times) if call_times else 0.0
    return {
        "video_id": vid,
        "events": events,
        "runtime_metadata": {
            "frames_processed": nframes,
            "chunks_processed": len(chunks),
            "end_to_end_internal_time_ms": round(ms, 1),
            "model_runtimes": [{
                "model_name": "qwen3-vl-4b-instruct-lora",
                "call_count": n,
                "total_time_ms": round(total_call_ms, 1),
                "average_time_ms": round(total_call_ms / n, 3),
                "p50_time_ms": round(_percentile(call_times, 0.50), 1),
                "p95_time_ms": round(_percentile(call_times, 0.95), 1),
                "max_time_ms": round(max(call_times), 1) if call_times else 0.0,
            }],
        },
    }


# --------------------------------------------------------------------------- #
def validate(preds, manifest_levels):
    errs = []
    seen = set()
    for p in preds:
        vid = p["video_id"]
        if vid in seen:
            errs.append(f"{vid}: duplicate")
        seen.add(vid)
        lvl = manifest_levels.get(vid)
        if lvl is None:
            errs.append(f"{vid}: not in manifest")
        for e in p["events"]:
            if e["class_name"] == "normal":
                errs.append(f"{vid}: class_name 'normal' is rejected, use events: []")
            if e["class_name"] not in CLASSES:
                errs.append(f"{vid}: bad class {e['class_name']}")
            if lvl == 1 and (e["start_time_sec"] is not None or e["end_time_sec"] is not None):
                errs.append(f"{vid}: L1 timestamps must be null")
            if lvl != 1:
                if e["start_time_sec"] is None or e["start_time_sec"] < 0:
                    errs.append(f"{vid}: start must be >= 0")
                elif e["end_time_sec"] is None or e["end_time_sec"] <= e["start_time_sec"]:
                    errs.append(f"{vid}: end must be > start")
            ex = e.get("explanation")
            if ex is not None and not (20 <= len(ex) <= 500):
                errs.append(f"{vid}: explanation length {len(ex)}")
        rm = p["runtime_metadata"]
        for mr in rm.get("model_runtimes", []):
            if {"total_time_ms", "call_count", "average_time_ms"} <= set(mr):
                exp = mr["total_time_ms"] / max(mr["call_count"], 1)
                if abs(exp - mr["average_time_ms"]) > 0.02 * max(exp, 1e-9):
                    errs.append(f"{vid}: average_time_ms mismatch")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--only", default="", help="comma-separated video ids")
    ap.add_argument("--levels", default="1,2,3")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    rows = man["videos"] if isinstance(man, dict) and "videos" in man else man
    levels = {r["video_id"]: int(r.get("level", 1)) for r in rows}
    files = {r["video_id"]: r.get("filename", r.get("file", r["video_id"] + ".mp4"))
             for r in rows}

    want_l = {int(x) for x in a.levels.split(",")}
    only = {x for x in a.only.split(",") if x}
    todo = [v for v in levels if levels[v] in want_l and (not only or v in only)]

    preds, t0 = [], time.perf_counter()
    with ThreadPoolExecutor(CFG["concurrency"]) as pool:
        for i, vid in enumerate(sorted(todo), 1):
            path = os.path.join(a.videos, files[vid])
            if not os.path.exists(path):
                print(f"!! missing {path}"); continue
            p = do_video(vid, path, levels[vid], pool)
            preds.append(p)
            print(f"[{i}/{len(todo)}] {vid} L{levels[vid]} -> "
                  f"{[e['class_name'] for e in p['events']] or 'normal'} "
                  f"({p['runtime_metadata']['end_to_end_internal_time_ms']:.0f} ms)")

    sub = {
        "schema_version": "1.0",
        "submission_id": f"run-{int(time.time())}",
        "model_name": "qwen3vl4b-lora-chunked",
        "run_metadata": {
            "total_wall_time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "max_parallel_videos": 1,   # videos run sequentially; chunks within a video are parallelised
            "hardware": os.environ.get("VAD_HW", "1x RTX 4090"),
        },
        "predictions": preds,
    }
    errs = validate(preds, levels)
    if errs:
        print("\nVALIDATION FAILED:"); [print(" -", e) for e in errs[:40]]
        return
    json.dump(sub, open(a.out, "w"), indent=1)
    kb = os.path.getsize(a.out) / 1024
    print(f"\nOK -> {a.out} ({kb:.0f} KB, {len(preds)} videos)")


if __name__ == "__main__":
    main()
