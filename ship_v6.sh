#!/bin/bash
# Merge -> serve -> full 34-video run -> score, for ONE checkpoint.
# One command, because the window between a checkpoint landing and the
# submission deadline is minutes.
#
#   ./ship_v6.sh out5/v0-.../checkpoint-165
#
# Writes submission_ft_v6_<step>.json and appends a row to
# checkpoint_results.jsonl (L1/L2/L3, class accuracy, precision, recall, marks)
# so checkpoints can be compared before deciding which to submit.
set -euo pipefail
cd "$(dirname "$0")"

CKPT="${1:?usage: ship_v6.sh <checkpoint-dir>}"
STEP="$(basename "$CKPT" | sed 's/[^0-9]//g')"
PORT=28451

export LD_LIBRARY_PATH=/home/miniorange/.local/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export MAX_PIXELS=200704 VIDEO_MAX_PIXELS=200704 CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0

# free the port: a server from a previous checkpoint is still holding VRAM
pkill -f "vllm serve .*-merged" 2>/dev/null || true
sleep 5

MERGED="${CKPT}-merged"
if [ ! -d "$MERGED" ]; then
  echo "== merging LoRA -> $(basename "$MERGED")"
  swift export --adapters "$CKPT" --merge_lora true >"merge_${STEP}.log" 2>&1
fi
[ -d "$MERGED" ] || { echo "merge failed, see merge_${STEP}.log"; tail -5 "merge_${STEP}.log"; exit 1; }

echo "== serving $(basename "$MERGED")"
nohup vllm serve "$MERGED" --served-model-name vad-qwen3vl-4b --port $PORT \
  --max-model-len 8192 --limit-mm-per-prompt '{"image":16}' --dtype bfloat16 \
  --gpu-memory-utilization 0.40 > "vllm_v6_${STEP}.log" 2>&1 &

for i in $(seq 1 90); do
  curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 4
done
curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null \
  || { echo "server never came up, see vllm_v6_${STEP}.log"; exit 1; }
echo "== server ready"

VAD_SERVER="http://127.0.0.1:$PORT/v1/chat/completions" \
  python3 build_v6.py --out "submission_ft_v6_${STEP}.json" \
                      --tag "checkpoint-${STEP}" \
                      --model_name "qwen3vl4b-lora-rebalanced-${STEP}"
