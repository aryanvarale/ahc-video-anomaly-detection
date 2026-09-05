"""
Live demo server for the drone anomaly detector.

Judges drop in an image or a video; the same pipeline that produces our
submissions runs against it and the per-chunk verdicts stream back over SSE as
they land, so the analysis is visible while it happens rather than appearing all
at once at the end.

    python webapp/server.py            # then open http://localhost:8420

Requires the vLLM server to be up (see README). If it isn't, the UI shows the
model as offline instead of failing silently.
"""
import base64
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vad_pipeline as vp  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
STATIC = APP_DIR / "static"
UPLOADS = APP_DIR / "uploads"
UPLOADS.mkdir(exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB_W = 220           # preview strip sent with each chunk verdict
MAX_UPLOAD_MB = 500

app = FastAPI(title="Drone Anomaly Detection — Live Demo")

# job_id -> {"path", "kind", "queue", "started"}
JOBS = {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def thumb(frame, width=THUMB_W):
    h, w = frame.shape[:2]
    s = width / max(w, 1)
    if s < 1.0:
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 62])
    return base64.b64encode(buf).decode() if ok else None


def top_classes(probs, n=4):
    return [{"cls": c, "p": round(p, 4)}
            for c, p in sorted(probs.items(), key=lambda x: -x[1])[:n]]


def model_online():
    base = vp.CFG["server"].rsplit("/v1/", 1)[0]
    try:
        r = requests.get(f"{base}/v1/models", timeout=2)
        if r.ok:
            ids = [m.get("id") for m in r.json().get("data", [])]
            return True, (ids[0] if ids else vp.CFG["model"])
    except Exception:
        pass
    return False, None


# --------------------------------------------------------------------------- #
# analysis workers - push progress dicts onto a queue consumed by the SSE route
# --------------------------------------------------------------------------- #
def analyse_image(path, q):
    frame = cv2.imread(str(path))
    if frame is None:
        q.put({"type": "error", "message": "Could not decode that image."})
        return

    q.put({"type": "meta", "kind": "image", "duration": 0.0,
           "chunks": 1, "frames": 1, "preview": thumb(frame, 640)})

    t0 = time.perf_counter()
    img = vp.encode(frame, vp.CFG["long_side"])
    probs = vp.classify([img]) if img else {"normal": 1.0}
    ms = (time.perf_counter() - t0) * 1000.0

    p_anom = 1.0 - probs.get("normal", 0.0)
    q.put({"type": "chunk", "idx": 0, "start": 0.0, "end": 0.0,
           "p_anom": round(p_anom, 4), "top": top_classes(probs),
           "ms": round(ms, 1), "thumb": thumb(frame)})

    if p_anom < vp.CFG["t_level1"]:
        events = []
    else:
        cls = max((c for c in probs if c != "normal"), key=lambda c: probs[c])
        events = [{"class_name": cls, "start_time_sec": None, "end_time_sec": None,
                   "confidence": round(probs[cls], 4),
                   "explanation": vp.EXPL.get(cls, "")}]

    q.put({"type": "events", "events": events})
    q.put({"type": "done", "runtime": {"total_ms": round(ms, 1), "calls": 1,
                                       "p50_ms": round(ms, 1), "rtf": None}})


def analyse_video(path, q):
    level = 2                      # demo uses the Level-2 chunking profile
    chunks, dur, nframes = vp.read_chunks(str(path), level)
    if not chunks:
        q.put({"type": "error", "message": "No frames could be read from that video."})
        return

    cap = cv2.VideoCapture(str(path))
    ok, first = cap.read()
    cap.release()

    q.put({"type": "meta", "kind": "video", "duration": round(dur, 2),
           "chunks": len(chunks), "frames": nframes,
           "preview": thumb(first, 640) if ok else None,
           "windows": [[round(a, 2), round(b, 2)] for a, b, _ in chunks]})

    call_times, probs_all = [], []
    wall0 = time.perf_counter()

    # bounded concurrency, but emitted in order so the sweep reads left-to-right
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(vp.CFG["concurrency"]) as pool:
        futures = [pool.submit(vp._timed_classify, c[2]) for c in chunks]
        for i, fut in enumerate(futures):
            probs, ms = fut.result()
            probs_all.append(probs)
            call_times.append(ms)
            a, b, imgs = chunks[i]
            p_anom = 1.0 - probs.get("normal", 0.0)
            payload = {"type": "chunk", "idx": i,
                       "start": round(a, 2), "end": round(b, 2),
                       "p_anom": round(p_anom, 4), "top": top_classes(probs),
                       "ms": round(ms, 1)}
            # mid-chunk frame as the preview for this window
            if imgs:
                payload["thumb_b64"] = imgs[len(imgs) // 2]
            q.put(payload)

    wall = (time.perf_counter() - wall0) * 1000.0
    events = [{"class_name": c, "start_time_sec": s, "end_time_sec": e,
               "confidence": round(score, 4), "explanation": vp.EXPL.get(c, "")}
              for c, s, e, score in vp.aggregate(chunks, probs_all)]

    q.put({"type": "events", "events": events})
    st = sorted(call_times)
    q.put({"type": "done", "runtime": {
        "total_ms": round(wall, 1),
        "calls": len(call_times),
        "p50_ms": round(vp._percentile(st, 0.50), 1),
        "p95_ms": round(vp._percentile(st, 0.95), 1),
        "rtf": round((wall / 1000.0) / dur, 3) if dur else None,
    }})


def run_job(job_id):
    job = JOBS[job_id]
    q = job["queue"]
    try:
        online, _ = model_online()
        if not online:
            q.put({"type": "error",
                   "message": "Model server is offline. Start vLLM and retry."})
            return
        (analyse_image if job["kind"] == "image" else analyse_video)(job["path"], q)
    except Exception as exc:                                   # noqa: BLE001
        q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        q.put({"type": "eof"})


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    online, model = model_online()
    return {"model_online": online, "model": model or vp.CFG["model"],
            "classes": vp.CLASSES,
            "thresholds": {"t_high": vp.CFG["t_high"], "t_low": vp.CFG["t_low"],
                           "t_video": vp.CFG["t_video"], "t_level1": vp.CFG["t_level1"]}}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    kind = "image" if ext in IMAGE_EXT else "video"
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job_id}{ext or '.mp4'}"

    size = 0
    with open(dest, "wb") as fh:
        while buf := await file.read(1 << 20):
            size += len(buf)
            if size > MAX_UPLOAD_MB * (1 << 20):
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB.")
            fh.write(buf)

    JOBS[job_id] = {"path": dest, "kind": kind, "queue": queue.Queue(),
                    "started": False}
    return {"job_id": job_id, "kind": kind, "name": file.filename, "bytes": size}


@app.get("/api/media/{job_id}")
def media(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return FileResponse(job["path"])


@app.get("/api/stream/{job_id}")
def stream(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if not job["started"]:
        job["started"] = True
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()

    def gen():
        q = job["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if msg.get("type") == "eof":
                yield "event: eof\ndata: {}\n\n"
                return
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8420"))
    print(f"\n  Drone anomaly detection demo -> http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
