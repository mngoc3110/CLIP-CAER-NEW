#!/bin/bash
set -e

# ====== RESUME CHECKPOINT (epoch ~15 best) ======
RESUME_CKPT="/kaggle/input/clip-caer-new-v1/CLIP-CAER-NEW-V1/model_best.pth"

python main.py \
  --mode train \
  --resume-from "${RESUME_CKPT}" \
  --exper-name stage2_sampler_LSR2 \
  --gpu 0 \
  --epochs 50 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.0001 \
  --milestones 20 35 \
  --gamma 0.1 \
  \
  --root-dir /kaggle/input/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
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
  --crop-body \
  \
  --label-smoothing 0.02 \
  --use-weighted-sampler \
  --lambda_mi 0.05 \
  --mi-warmup 10 \
  --mi-ramp 15 \
  --lambda_dc 0.08 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  \
  --use-amp \
  --grad-clip 1.0 \
  --seed 42 \
  --print-freq 10