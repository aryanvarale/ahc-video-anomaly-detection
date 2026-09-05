#!/bin/bash
# ~30 minute corrective pass: take the SHIPPED adapter and keep training it on the
# class-balanced data at a low learning rate.
#
# A full balanced retrain is 3h20m and there is one hour. Continuing the existing
# adapter instead of starting over keeps everything it already learned and only
# asks it to unlearn the class prior - on held-out video the old data's 8:1
# anomaly imbalance shows up as a traffic_accident magnet pulling 18 chunks in
# from five other classes. Re-weighting a prior is fast; it does not need an epoch.
#
# resume_only_model=true loads the adapter WITHOUT the old optimizer/scheduler
# state, so this is a fresh short schedule rather than a resumption of step 2499.
# lr is 3e-5 rather than the original 1e-4 because the model is already converged
# and the goal is a nudge, not another fit.
set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/home/miniorange/.local/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export MAX_PIXELS=200704
export VIDEO_MAX_PIXELS=200704
export CUDA_VISIBLE_DEVICES=0

python3 /home/miniorange/.local/lib/python3.10/site-packages/swift/cli/sft.py \
  --model ./qwen3vl4b --model_type qwen3_vl \
  --tuner_type lora \
  --adapters out3/v1-20260905-013645/checkpoint-2499 \
  --dataset sft_bal_train.jsonl \
  --freeze_vit true \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 \
  --max_steps 330 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.03 \
  --save_steps 165 \
  --eval_strategy no \
  --logging_steps 10 \
  --output_dir out5
