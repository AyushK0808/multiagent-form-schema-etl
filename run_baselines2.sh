#!/bin/bash
# Remaining frozen-baseline runs: Group 1 for the other two seeds (Group 2 baseline is one seed, for cost).
cd "$(dirname "$0")"
export PATH="$PATH:/c/Program Files/Tesseract-OCR" PYTHONIOENCODING=utf-8
while tasklist | grep -q " 28604 "; do sleep 30; done
for spec in "13:models/baseline_frozen_seed13" "1234:models/baseline_frozen_seed1234"; do
  seed=${spec%%:*}; root=${spec#*:}
  .venv/Scripts/python.exe finetune/train_lora.py --baseline frozen --groups group_1 \
    --no-augment --learning-rate 1e-3 --seed "$seed" --output-root "$root" >> baseline_train.log 2>&1
done
for run in models/baseline_frozen/group_1 models/baseline_frozen/group_2 models/baseline_frozen_seed13/group_1 models/baseline_frozen_seed1234/group_1; do
  .venv/Scripts/python.exe finetune/eval_test.py --runs $run/layoutlmv3 >> eval_test_baseline.log 2>&1
done
echo ALL_DONE >> baseline_train.log
