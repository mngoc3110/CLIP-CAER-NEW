#!/bin/bash

# ======================================================================================
# This script is configured for LOCAL training.
#
# PLEASE UPDATE THE FOLLOWING PATHS to match your local machine's setup.
# You need to provide the absolute or relative paths to your dataset folders.
# ======================================================================================

# --- PATHS TO CONFIGURE ---
ROOT_DIR="./" # e.g., /Users/macbook/Desktop/CLIP-CAER/RAER
ANNOT_DIR="./" # e.g., /Users/macbook/Desktop/CLIP-CAER/annotations

# --- SCRIPT ---
python main.py \
    --mode train \
    --exper-name local_m2max_safe_end_to_end \
    --gpu mps \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-5 \
    --lr-image-encoder 0.0 \
    --lr-prompt-learner 1e-5 \
    --lr-adapter 1e-5 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --milestones 20 35 \
    --gamma 0.1 \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --print-freq 10 \
    --root-dir "$ROOT_DIR" \
    --train-annotation "$ANNOT_DIR/RAER/annotation/train_80.txt" \
    --val-annotation "$ANNOT_DIR/RAER/annotation/val_20.txt" \
    --test-annotation "$ANNOT_DIR/RAER/annotation/test.txt" \
    --clip-path ViT-B/32 \
    --bounding-box-face "$ANNOT_DIR/RAER/bounding_box/face.json" \
    --bounding-box-body "$ANNOT_DIR/RAER/bounding_box/body.json" \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --lambda_mi 0.2 \
    --lambda_dc 0.3 \
    --mi-warmup 10 \
    --mi-ramp 10 \
    --dc-warmup 10 \
    --dc-ramp 15 \
    --slerp-weight 0.5 \
    --temperature 0.2 \
    --use-weighted-sampler \
    --label-smoothing 0.02 \
    --use-amp \
    --crop-body \
    --grad-clip 1.0