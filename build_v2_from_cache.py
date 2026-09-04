import json
import vad_pipeline as vp

cache = json.load(open("chunk_cache.json"))
v1 = json.load(open("submission_v1_zeroshot.json"))
rmeta = {p["video_id"]: p["runtime_metadata"] for p in v1["predictions"]}

preds = []
for vid, entry in sorted(cache.items()):
    level = entry["level"]
    chunks = [(w[0], w[1], None) for w in entry["windows"]]
    probs = entry["probs"]

    if level == 1:
        agg = {}
        for p in probs:
            for c, v in p.items():
                agg[c] = agg.get(c, 0.0) + v
        z = sum(agg.values()) or 1.0
        agg = {k: v / z for k, v in agg.items()}
        p_anom = 1.0 - agg.get("normal", 0.0)
        if p_anom < vp.CFG["t_level1"]:
            events = []
        else:
            cls = max((c for c in agg if c != "normal"), key=lambda c: agg[c])
            events = [{"class_name": cls, "start_time_sec": None, "end_time_sec": None,
                       "explanation": vp.EXPL[cls]}]
    else:
        events = [{"class_name": c, "start_time_sec": s, "end_time_sec": e,
                   "explanation": vp.EXPL[c]} for c, s, e, _ in vp.aggregate(chunks, probs)]

    preds.append({"video_id": vid, "events": events, "runtime_metadata": rmeta[vid]})

sub = {
    "schema_version": "1.0",
    "submission_id": "run-v2-tuned",
    "model_name": "qwen3vl4b-zeroshot-tuned",
    "run_metadata": v1["run_metadata"],
    "predictions": preds,
}

errs = vp.validate(preds, {p["video_id"]: (1 if not p["events"] or p["events"][0]["start_time_sec"] is None else 2)
                            for p in preds})
# proper level lookup instead of guessing from events
man = json.load(open("manifest.json"))
levels = {r["video_id"]: int(r["level"]) for r in man["videos"]}
errs = vp.validate(preds, levels)
if errs:
    print("VALIDATION FAILED:")
    for e in errs[:40]:
        print(" -", e)
else:
    json.dump(sub, open("submission_v2_tuned.json", "w"), indent=1)
    print("OK -> submission_v2_tuned.json")
