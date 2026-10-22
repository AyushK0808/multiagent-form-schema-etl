#!/bin/bash
# Frozen-backbone baseline (classifier head only), same data/seeds as the LoRA runs.
cd "$(dirname "$0")"
for spec in "42:models/baseline_frozen" "13:models/baseline_frozen_seed13" "1234:models/baseline_frozen_seed1234"; do
  seed=${spec%%:*}; root=${spec#*:}
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe finetune/train_lora.py --baseline frozen \
    --no-augment --learning-rate 1e-3 --seed "$seed" --output-root "$root" >> "baseline_train.log" 2>&1
done
echo ALL_DONE >> baseline_train.log
