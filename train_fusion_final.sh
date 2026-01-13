#!/bin/bash
set -e

# Config for Option A: Prioritize maximizing overall UAR (stable, no collapse)
# This script uses the stable "probability fusion" (Mixture-of-Experts) method.

# NOTE: Please replace the placeholder path for --resume-from.
RESUME_CKPT="/kaggle/input/clip-caer-new-v1/CLIP-CAER-NEW-V1/model_best.pth"

python main.py \
  --mode train \
  --resume-from "${RESUME_CKPT}" \
  --reset-epoch \
  --exper-name UAR_max_fusion_option_A \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-4 \
  --weight-decay 0.0001 \
  \
  --two-head-loss \
  --w-bin 1.2 \
  --w-4 1.0 \
  --sweep-range 0.25 0.85 0.02 \
  --conf-recall-min 0.0 \
  \
  --root-dir /kaggle/input/raer-video-emotion-dataset/ \
  --train-annotation /kaggle/input/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json \
  \

  --lambda_mi 0.03 \
  --mi-warmup 2 \
  --mi-ramp 5 \
  --lambda_dc 0.05 \
  --dc-warmup 2 \
  --dc-ramp 5 \
  \
  --use-amp \
  --grad-clip 1.0 \
  --seed 42 \
  --print-freq 10 \
  --temporal-layers 1 \
  --contexts-number 12 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body False \
  --lr-adapter 5e-5 \
  --momentum 0.9 \
  --milestones 20 35 \
  --gamma 0.1 \
  --slerp-weight 0.0