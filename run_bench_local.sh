#!/bin/bash
# Fully local vision backend (Ollama): no daily quota, so full test sets are feasible.
# Order: synthetic, SROIE, then CORD last (slowest, most likely to hit model loops).
# Subsets are seeded, so raising --limit later extends them without changing the first documents.
cd "$(dirname "$0")"
export PATH="$PATH:/c/Program Files/Tesseract-OCR" PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 VISION_BACKEND=ollama OLLAMA_VISION_MODEL=${OLLAMA_VISION_MODEL:-qwen2.5vl:7b}
PY=.venv/Scripts/python.exe
echo "vision model: $OLLAMA_VISION_MODEL"
$PY -m eval.run_benchmark --dataset synth --out results_local 2>&1 | tee -a bench_local_synth.log
$PY -m eval.run_benchmark --dataset sroie --limit 100 --out results_local 2>&1 | tee -a bench_local_sroie.log
$PY -m eval.run_benchmark --dataset cord --limit 50 --out results_local 2>&1 | tee -a bench_local_cord.log
echo ALL_DONE | tee -a bench_local_cord.log
