"""
Build LoRA SFT data for the AHC VAD hackathon.

Two sources, one output format (ms-swift messages+images JSONL):

 1. LABELLED benchmarks (UCF-Crime, XD-Violence, DoTA/DADA, Drone-Anomaly...):
    map their annotations onto our 12 classes and cut chunks directly.
    --> free, exact, do this FIRST.

 2. UNLABELLED drone footage: auto-label 2s chunks with a larger LOCAL open
    VLM (Qwen2.5-VL-7B-Instruct by default), run on-GPU with no API key and
    no per-call cost. Allowed by the rules for training-data generation since
    it never touches the runtime detector. Keeps only chunks where two
    independent passes with different frame orderings agree.

Then it BALANCES the set so ~45% of samples are `normal`, drawn preferentially
from hard negatives. False alarms are the single most expensive failure mode in
the scoring, so the normal class must be well represented.

Output line:
{"messages":[{"role":"user","content":"<image>...<image>\n<PROMPT>"},
             {"role":"assistant","content":"C"}],
 "images":["/abs/f0.jpg", ...]}
"""

import argparse, glob, json, os, random, re
from collections import Counter

import cv2
import torch

from vad_pipeline import CLASSES, LETTERS, PROMPT, probe

C2L = dict(zip(CLASSES, LETTERS))
TEACHER_MODEL_PATH = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

_teacher, _processor = None, None


def _load_teacher():
    """Lazily load the local teacher VLM once, kept resident for the run."""
    global _teacher, _processor
    if _teacher is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"[teacher] loading {TEACHER_MODEL_PATH} ...")
        _teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            TEACHER_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        _processor = AutoProcessor.from_pretrained(TEACHER_MODEL_PATH)
    return _teacher, _processor

# map public-benchmark labels -> our 12 classes. Extend as you inspect the data.
BENCH_MAP = {
    "RoadAccidents": "traffic_accident", "Accident": "traffic_accident",
    "Fighting": "fighting_or_violence", "Assault": "fighting_or_violence",
    "Abuse": "fighting_or_violence", "Riot": "fighting_or_violence",
    "Explosion": "fire", "Fire": "fire", "Arson": "fire",
    "Normal": "normal", "Normal_Videos_event": "normal", "Testing_Normal": "normal",
}


def cut_chunks(path, chunk_sec=2.0, stride=2.0, k=8, long_side=448, outdir="chunks"):
    """Write jpeg frames for each chunk; return [(start,end,[paths])]."""
    fps, n, dur = probe(path)
    base = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(outdir, exist_ok=True)
    res, t = [], 0.0
    cap = cv2.VideoCapture(path)
    frames = {}
    want = []
    while t < dur - 0.5:
        idxs = [int(round((t + chunk_sec * (j + 0.5) / k) * fps)) for j in range(k)]
        want.append((t, min(t + chunk_sec, dur), [min(max(i, 0), n - 1) for i in idxs]))
        t += stride
    need = sorted({i for _, _, ii in want for i in ii})
    ptr, i = 0, 0
    while ptr < len(need):
        if not cap.grab():
            break
        if i == need[ptr]:
            ok, fr = cap.retrieve()
            if ok:
                # resize IMMEDIATELY - buffering full-res frames blows up host RAM
                h, w = fr.shape[:2]
                s = long_side / max(h, w)
                if s < 1.0:
                    fr = cv2.resize(fr, (int(w * s) // 28 * 28, int(h * s) // 28 * 28))
                frames[i] = fr
            ptr += 1
            while ptr < len(need) and need[ptr] == i:
                ptr += 1
        i += 1
    cap.release()
    for ci, (a, b, idxs) in enumerate(want):
        paths = []
        for j, ix in enumerate(idxs):
            if ix not in frames:
                continue
            p = os.path.abspath(f"{outdir}/{base}_c{ci:04d}_{j}.jpg")
            cv2.imwrite(p, frames[ix], [cv2.IMWRITE_JPEG_QUALITY, 88])
            paths.append(p)
        if len(paths) >= k - 1:
            res.append((a, b, paths))
    return res


def ask_teacher(paths):
    from qwen_vl_utils import process_vision_info
    model, processor = _load_teacher()
    content = [{"type": "image", "image": f"file://{p}"} for p in paths]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    out_ids = gen[:, inputs["input_ids"].shape[1]:]
    txt = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip().upper()
    m = re.search(r"[A-L]", txt)
    return m.group(0) if m else None


def label_one(item):
    _, _, paths = item
    a = ask_teacher(paths)
    b = ask_teacher(list(reversed(paths)))      # cheap self-consistency check
    return a if (a and a == b) else None


def resolve_class(path):
    """Class from directory layout. Handles both
    '<class_name>/videos/x.mp4' (this hackathon's own dataset, folder name
    IS the class) and generic benchmark layouts like 'UCF/<ClassName>/x.mp4'
    (mapped via BENCH_MAP)."""
    parent = os.path.basename(os.path.dirname(path))
    if parent.lower() == "videos":
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if parent in CLASSES:
        return parent
    return BENCH_MAP.get(parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabelled", default="", help="glob of drone/CCTV videos")
    ap.add_argument("--labelled", default=[], action="append",
                     help="glob like 'Train and Test/train/*/videos/*.mp4' or 'UCF/*/*.mp4'; "
                          "pass multiple times to combine several sources into one balanced set")
    ap.add_argument("--chunkdir", default="chunks")
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--max_per_video", type=int, default=12)
    ap.add_argument("--max_videos_per_class", type=int, default=100,
                     help="cap source videos per class before chunk-cutting, so a "
                          "class with 10x more raw videos doesn't dominate decode time")
    ap.add_argument("--normal_share", type=float, default=0.45)
    a = ap.parse_args()

    rows = []

    # ---- source 1: labelled benchmarks (no teacher calls needed) ----
    by_class_paths = {}
    for pattern in a.labelled:
        for path in glob.glob(pattern):
            cls = resolve_class(path)
            if not cls:
                continue
            by_class_paths.setdefault(cls, []).append(path)

    for cls, paths in by_class_paths.items():
        random.shuffle(paths)
        capped = paths[: a.max_videos_per_class]
        print(f"[bench] {cls}: {len(paths)} found, using {len(capped)}")
        for path in capped:
            ch = cut_chunks(path, outdir=a.chunkdir)
            random.shuffle(ch)
            for _, _, chpaths in ch[: a.max_per_video]:
                rows.append((C2L[cls], chpaths))

    # ---- source 2: unlabelled footage, distilled from a large VLM ----
    todo = []
    for path in glob.glob(a.unlabelled):
        ch = cut_chunks(path, outdir=a.chunkdir)
        random.shuffle(ch)
        todo += ch[: a.max_per_video]
    print(f"[teacher] labelling {len(todo)} chunks with local {TEACHER_MODEL_PATH}")
    if todo:
        _load_teacher()  # load once, up front, before the loop
        for i, item in enumerate(todo, 1):
            lab = label_one(item)
            if lab:
                rows.append((lab, item[2]))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")

    # ---- balance ----
    # normal_share is a target, not just a normal-side cap: if there isn't enough
    # normal supply to hit it against the full anomaly pool (false alarms are the
    # costliest failure mode, so under-shooting this silently is dangerous),
    # downsample the anomaly side proportionally instead of quietly accepting a
    # lower normal share.
    by = {}
    for lab, paths in rows:
        by.setdefault(lab, []).append(paths)

    norm_pool = by.get("A", [])[:]
    random.shuffle(norm_pool)
    anom_labels = [k for k in by if k != "A"]
    n_anom_total = sum(len(by[k]) for k in anom_labels)
    n_norm_avail = len(norm_pool)

    keep_norm = int(n_anom_total * a.normal_share / max(1 - a.normal_share, 1e-6))
    if n_norm_avail >= keep_norm:
        final_norm = norm_pool[:keep_norm]
        final_anom_by_class = {k: by[k] for k in anom_labels}
    else:
        max_anom_total = int(n_norm_avail * (1 - a.normal_share) / max(a.normal_share, 1e-6))
        final_norm = norm_pool
        final_anom_by_class = {}
        if 0 < max_anom_total < n_anom_total:
            frac = max_anom_total / n_anom_total
            for k in anom_labels:
                v = by[k][:]
                random.shuffle(v)
                final_anom_by_class[k] = v[: max(1, round(len(v) * frac))]
        else:
            final_anom_by_class = {k: by[k] for k in anom_labels}

    final = [("A", p) for p in final_norm]
    for k, v in final_anom_by_class.items():
        final += [(k, p) for p in v]
    random.shuffle(final)

    with open(a.out, "w") as f:
        for lab, paths in final:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "<image>" * len(paths) + "\n" + PROMPT},
                    {"role": "assistant", "content": lab},
                ],
                "images": paths,
            }) + "\n")
    print("\nclass counts:", Counter(l for l, _ in final))
    print(f"wrote {len(final)} samples -> {a.out}")


if __name__ == "__main__":
    main()
