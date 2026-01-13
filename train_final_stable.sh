#!/bin/bash
set -e

# FINAL STABLE SCRIPT
# This script uses the stable "probability fusion" method and disables AMP to prevent 'inf' loss.
# All code files have been synchronized to fix the size mismatch error.

# NOTE: Please replace the placeholder path for --resume-from.
RESUME_CKPT="/PATH/TO/YOUR/model_best.pth"

python main.py \
  --mode train \
  --resume-from "${RESUME_CKPT}" \
  --reset-epoch \
  --exper-name final_stable_fusion_run \
  --gpu 0 \
  --epochs 15 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-4 \
  --lr-adapter 5e-5 \
  --weight-decay 0.0001 \
  --two-head-loss \
  --w-bin 0.5 \
  --w-4 0.5 \
  --root-dir /kaggle/input/raer-video-emotion-dataset/ \
  --train-annotation /kaggle/input/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --clip-path ViT-B/32 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --grad-clip 1.0 \
  --seed 42 \
  --print-freq 10