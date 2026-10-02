#!/usr/bin/env bash
# ==============================================================================
# Full Training and Evaluation Pipeline for Tashkeel-v3 SOTA Model
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "========================================================================"
echo "STARTING FULL TASHKEEL-v3 PIPELINE"
echo "========================================================================"

# Stage 1: Build & Deduplicate Multi-Corpus
echo "[Step 1/5] Building deduplicated multi-corpus..."
python 01_build_multicorpus.py

# Stage 2: Train Morpheme Backbone with DDP
echo "[Step 2/5] Training Morpheme-Aware MARBERTv2 Backbone..."
torchrun --nproc_per_node=2 02_train_morpheme_backbone.py --batch-size 128 --epochs 3

# Stage 3: Domain Calibration
echo "[Step 3/5] Domain Calibration Fine-Tuning..."
python 03_domain_calibration.py

# Stage 4: Train Linear-Chain CRF Layer
echo "[Step 4/5] Training Linear-Chain CRF with Viterbi Sequence Optimization..."
torchrun --nproc_per_node=2 04_train_crf_viterbi.py --batch-size 128 --epochs 2

# Stage 5: Evaluate on Benchmarks
echo "[Step 5/5] Running Official Benchmark Evaluations..."
python 05_evaluate_benchmarks.py --ckpt-path ./checkpoints/stage3_crf_final/best_model.pt

# Export Package
echo "Exporting Hugging Face deployment package..."
python 06_export_hf_package.py

echo "========================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================================================"
