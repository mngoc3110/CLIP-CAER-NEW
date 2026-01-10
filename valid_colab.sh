#!/bin/bash

python main.py \
    --mode eval \
    --gpu 0 \
    --exper-name test_eval_colab \
    --eval-checkpoint /content/drive/MyDrive/khoaluan/Dataset/outputs/train_baseline-[12-29]-[10:50]/model_best.pth \
    --root-dir /content/drive/MyDrive/khoaluan/Dataset/RAER \
    --test-annotation /content/drive/MyDrive/khoaluan/Dataset/raer-annot/annotation/test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face /content/drive/MyDrive/khoaluan/Dataset/raer-annot/annotation/bounding_box/face.json \
    --bounding-box-body /content/drive/MyDrive/khoaluan/Dataset/raer-annot/annotation/bounding_box/body.json \
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
