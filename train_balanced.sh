#!/bin/bash
# Retrain the LoRA on the class-balanced set.
#
# The previous run's data was ~8:1 between the commonest and rarest anomaly
# (loitering 1621 chunks vs waterlogging 193), which is exactly the shape that
# makes a model fall back on the frequent class when unsure - the observed
# fighting->loitering and {congestion,wrong_way,spill}->traffic_accident errors.
# sft_bal.jsonl equalises every anomaly class at 650 chunks.
#
# Unlike the previous run this holds out 8% of the *videos* and passes them as a
# validation set, so generalisation is measured while training rather than
# inferred from the final train accuracy.
set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/home/miniorange/.local/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export MAX_PIXELS=200704
export VIDEO_MAX_PIXELS=200704
export CUDA_VISIBLE_DEVICES=0

python3 /home/miniorange/.local/lib/python3.10/site-packages/swift/cli/sft.py \
  --model ./qwen3vl4b --model_type qwen3_vl \
  --tuner_type lora \
  --dataset sft_bal_train.jsonl \
  --val_dataset sft_bal_val.jsonl \
  --freeze_vit true \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --save_steps 200 \
  --eval_strategy steps \
  --eval_steps 200 \
  --per_device_eval_batch_size 1 \
  --logging_steps 10 \
  --output_dir out4
