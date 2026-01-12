#!/bin/bash
set -e

RESUME_CKPT="/kaggle/input/clip-caer-new-v1/CLIP-CAER-NEW-V1/model_best.pth"

python main.py \
  --mode train \
  --resume-from "${RESUME_CKPT}" \
  --exper-name stage2_conf_split_softgate \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 5e-5 \
  --weight-decay 1e-4 \
  \
  --milestones 20 23 \
  --gamma 0.1 \
  \
  --two-head-loss \
  --w-bin 1.5 \
  --w-4 1.0 \
  --soft-gate-thr 0.7 \
  \
  --lambda-mi 0.03 \
  --mi-warmup 16 \
  --mi-ramp 3 \
  \
  --lambda-dc 0.05 \
  --dc-warmup 16 \
  --dc-ramp 3 \
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
  --use-amp \
  --grad-clip 1.0 \
  --seed 42 \
  --print-freq 10 \
  \
  --root-dir /kaggle/input/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json