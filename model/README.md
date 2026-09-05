# LoRA adapter

`adapter_config.json` is the exact config of the checkpoint used to produce the
submitted results (rank 16, alpha 32, `all-linear`, frozen ViT).

The trained weights (`adapter_model.safetensors`, ~132MB) are not committed —
GitHub rejects files over 100MB without Git LFS. Two ways to get them:

1. **Retrain** — `bash train.sh` from the repo root reproduces this checkpoint
   from the same data and hyperparameters (a few hours on one GPU).
2. **Download** — ask the maintainer for the weights file and drop it next to
   this config as `adapter_model.safetensors`, then:

   ```bash
   swift export --adapters ./model --merge_lora true
   ```
