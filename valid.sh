#!/bin/bash

python main.py \
    --mode eval \
    --gpu 2 \
    --exper-name test_eval \
    --eval-checkpoint outputs/m2max_fastUAR70_vitb32_wrs_logitadj_tau05_mi07_dc12/model_best.pth \
    --root-dir /kaggle/input/raer-video-emotion-dataset/RAER \
    --train-annotation /kaggle/input/raer-annot/annotation/train_80.txt \
    --val-annotation /kaggle/input/raer-annot/annotation/val_20.txt \
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
    --temperature 0.07 \
    --crop-body
