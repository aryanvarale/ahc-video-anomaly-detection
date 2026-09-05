#!/bin/bash
# LoRA SFT run that produced the shipped checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

export MAX_PIXELS=200704
export VIDEO_MAX_PIXELS=200704
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cu13,os;print(os.path.dirname(nvidia.cu13.__file__)+"/lib")' 2>/dev/null || true):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

swift sft \
  --model ./qwen3vl4b --model_type qwen3_vl --tuner_type lora \
  --dataset sft.jsonl \
  --freeze_vit true --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --warmup_ratio 0.05 \
  --save_steps 200 --logging_steps 10 --output_dir out
