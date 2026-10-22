#!/bin/bash
# One job at a time: the Groq free tier allows ~200k vision tokens/day, shared by all runs.
# Cached branches make this resumable; raise --limit later to extend a subset.
cd "$(dirname "$0")"
export PATH="$PATH:/c/Program Files/Tesseract-OCR" PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m eval.run_benchmark --dataset synth --out results_final 2>&1 | tee -a bench_synth.log
.venv/Scripts/python.exe -m eval.run_benchmark --dataset cord --limit 50 --out results_final 2>&1 | tee -a bench_cord.log
.venv/Scripts/python.exe -m eval.run_benchmark --dataset sroie --limit 50 --out results_final 2>&1 | tee -a bench_sroie.log
echo ALL_DONE | tee -a bench_sroie.log
