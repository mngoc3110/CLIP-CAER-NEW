#!/bin/bash
set -e

# STAGE 3: Recall-focused sweep.
# Changes:
# 1. Lowered threshold sweep range to be more sensitive to Confusion.
# 2. Enabled minimum recall constraint for Confusion class.

# NOTE: Please replace the placeholder path for --resume-from.
RESUME_CKPT="/kaggle/input/clip-caer-new-v1/CLIP-CAER-NEW-V1/model_best.pth"

python main.py \
  --mode train \
  --resume-from "${RESUME_CKPT}" \
  --reset-epoch \
  --exper-name stage3_recall_focus_sweep \
  --gpu 0 \
  --epochs 5 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 5e-5 \
  \
  --two-head-loss \
  --w-bin 2.5 \
  --w-4 1.0 \
  --sweep-range 0.05 0.21 0.05 \
  --conf-recall-min 40.0 \
  \
  --root-dir ./RAER \
  --train-annotation /kaggle/input/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --text-type prompt_ensemble \
  --clip-path ViT-B/32 \
  \
  --use-weighted-sampler \
  --lambda_mi 0.05 \
  --mi-warmup 2 \
  --mi-ramp 3 \
  --lambda_dc 0.08 \
  --dc-warmup 2 \
  --dc-ramp 3 \
  \
  --use-amp \
  --grad-clip 1.0 \
  --seed 42 \
  --print-freq 10
