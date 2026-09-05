"""
Build LoRA SFT data (ms-swift messages+images JSONL).

Two sources:
 1. Labelled videos: folder name is the class (or BENCH_MAP maps a public
    benchmark's label onto one of ours). Chunks cut directly, no model call.
 2. Unlabelled footage: auto-labelled by a local teacher VLM
    (Qwen2.5-VL-7B-Instruct), kept only if two passes with reversed frame
    order agree.

Then balanced: ~45% normal, anomaly classes optionally equalised via
--anom_target (denser stride for scarce classes, round-robin over source
videos so the cap buys variety, not just count).

Usage: python -m src.build_training_data --labelled 'train/*/videos/*.mp4' \
    --out sft.jsonl --anom_target 650
"""

import argparse, glob, json, os, random, re
from collections import Counter

import cv2
import torch

from src.pipeline import CLASSES, LETTERS, PROMPT, probe

C2L = dict(zip(CLASSES, LETTERS))
TEACHER_MODEL_PATH = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

_teacher, _processor = None, None


def _load_teacher():
    global _teacher, _processor
    if _teacher is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"[teacher] loading {TEACHER_MODEL_PATH} ...")
        _teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            TEACHER_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        _processor = AutoProcessor.from_pretrained(TEACHER_MODEL_PATH)
    return _teacher, _processor


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
    b = ask_teacher(list(reversed(paths)))
    return a if (a and a == b) else None


def resolve_class(path):
    """'<class>/videos/x.mp4' -> class directly; else BENCH_MAP lookup."""
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
                     help="glob, e.g. 'train/*/videos/*.mp4'; repeatable")
    ap.add_argument("--chunkdir", default="chunks")
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--max_per_video", type=int, default=12)
    ap.add_argument("--max_videos_per_class", type=int, default=100)
    ap.add_argument("--normal_share", type=float, default=0.45)
    ap.add_argument("--anom_target", type=int, default=0,
                     help="chunks per anomaly class (0 = natural counts)")
    a = ap.parse_args()

    rows = []

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

        stride, per_video = 2.0, a.max_per_video
        if a.anom_target and cls != "normal":
            need = a.anom_target / max(len(capped), 1)
            if need > 6:
                stride = 0.5
            elif need > 3:
                stride = 1.0
            per_video = max(1, min(a.max_per_video, int(need * 2)))

        print(f"[bench] {cls}: {len(paths)} found, using {len(capped)} "
              f"(stride {stride}, <={per_video}/video)", flush=True)
        for path in capped:
            ch = cut_chunks(path, stride=stride, outdir=a.chunkdir)
            random.shuffle(ch)
            for _, _, chpaths in ch[:per_video]:
                rows.append((C2L[cls], chpaths, path))

    todo = []
    for path in glob.glob(a.unlabelled):
        ch = cut_chunks(path, outdir=a.chunkdir)
        random.shuffle(ch)
        todo += [(c, path) for c in ch[: a.max_per_video]]
    print(f"[teacher] labelling {len(todo)} chunks with local {TEACHER_MODEL_PATH}")
    if todo:
        _load_teacher()
        for i, (item, src) in enumerate(todo, 1):
            lab = label_one(item)
            if lab:
                rows.append((lab, item[2], src))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")

    by = {}
    by_src = {}
    for lab, paths, src in rows:
        by.setdefault(lab, []).append(paths)
        by_src.setdefault(lab, {}).setdefault(src, []).append(paths)

    if a.anom_target:
        for k in list(by):
            if k == "A":
                continue
            groups = [v[:] for v in by_src[k].values()]
            for g in groups:
                random.shuffle(g)
            random.shuffle(groups)
            picked, i = [], 0
            while len(picked) < a.anom_target and any(groups):
                g = groups[i % len(groups)]
                if g:
                    picked.append(g.pop())
                else:
                    groups.pop(i % len(groups))
                    continue
                i += 1
            by[k] = picked

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
