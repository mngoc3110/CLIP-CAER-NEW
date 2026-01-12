#!/bin/bash
set -e

# STAGE 2 (v2) - Tuned configuration based on Epoch 16/17 results.
# Changes:
# 1. Re-enabled --use-weighted-sampler.
# 2. Adjusted soft-gate and binary loss weight (--soft-gate-thr 0.5, --w-bin 2.5).
# 3. Delayed MI/DC loss warmup to start after epoch 16.

# NOTE: Please replace the placeholder path for --resume-from.
RESUME_CKPT="/kaggle/input/clip-caer-new-v1/CLIP-CAER-NEW-V1/model_best.pth"

python main.py \
  --mode train \
  --resume-from "${RESUME_CKPT}" \
  --exper-name stage2_v2_sampler_tuned_gate_find_threst_hold \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 5e-5 \
  --weight-decay 0.0001 \
  --milestones 18 19 \
  --gamma 0.1 \
  \
  --two-head-loss \
  --w-bin 2.5 \
  --w-4 1.0 \
  --soft-gate-thr 0.5 \
  \
  --root-dir /kaggle/input/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/raer-video-emotion-dataset/RAER/annotation/train.txt \
  --val-annotation /kaggle/input/raer-video-emotion-dataset/RAER/annotation/test.txt \
  --test-annotation /kaggle/input/raer-video-emotion-dataset/RAER/annotation/test.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json \
  \
  --text-type prompt_ensemble \
  --contexts-number 12 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  \
  --use-weighted-sampler \
  --lambda_mi 0.05 \
  --mi-warmup 17 \
  --mi-ramp 3 \
  --lambda_dc 0.08 \
  --dc-warmup 17 \
  --dc-ramp 3 \
  \
  --use-amp \
  --grad-clip 1.0 \
  --seed 42 \
  --print-freq 10
