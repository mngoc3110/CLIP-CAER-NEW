#!/bin/bash

python main.py \
    --mode eval \
    --gpu 2 \
    --exper-name test_eval \
    --eval-checkpoint outputs/train_baseline-[12-29]-[10:50]/model_best.pth \
    --root-dir /kaggle/input/raer-video-emotion-dataset/RAER \
    --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face /kaggle/input/raer-annot/annotation/bounding_box/face.json \
    --bounding-box-body /kaggle/input/raer-annot/annotation/bounding_box/body.json \
    --text-type class_descriptor \
    --contexts-number 12 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --slerp-weight 0.5 \
    --temperature 0.07