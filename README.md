# CLIP-CAER with Advanced Training Strategies

This repo is the official implementation for CLIP-based Context-aware Academic Emotion Recognition[[arXiv](https://arxiv.org/abs/2507.00586)], with additional advanced training strategies implemented to further boost performance, including **Expression-Aware Adapters (EAA)**, **Instance-Enhanced Classifiers (IEC)**, and **Mutual Information (MI) Loss** for dual-view prompt regularization. The paper has been accepted to ICCV 2025.

## Introduction
In this paper, we propose CLIP-CAER, a context-aware academic emotion recognition method based on CLIP. By leveraging contextual information from learning scenarios, our method significantly improves the model’s ability to recognize students’ learning states (i.e., focused or distracted). Notably, it achieves an approximately 20% improvement in accuracy for the distraction category.<br>

This repository enhances the original CLIP-CAER by integrating several state-of-the-art techniques aimed at improving feature representation and handling data imbalance, targeting a UAR (Unweighted Average Recall) of over 70%.

![image-20250702095507999](./imgs/pipeline.png)

## New Architecture Details

The enhanced architecture builds upon the CLIP-CAER backbone, incorporating several new modules to create a more robust and accurate model.

### 1. Dual-Stream Visual Backbone
The model processes two separate visual streams:
- **Face Stream:** Cropped facial regions to capture fine-grained facial expressions.
- **Context Stream:** The full video frame or body region to capture surrounding context and behavior.

Both streams are processed by a shared CLIP Visual Encoder.

### 2. EAA (Expression-Aware Adapter)
To better capture subtle, emotion-specific facial details without sacrificing the generalization power of the pre-trained CLIP model, a lightweight **Expression-Aware Adapter** is integrated into the face stream.
- **Implementation:** A bottleneck adapter module is inserted after the CLIP visual encoder for the face stream.
- **Trainable:** Only the adapter's parameters are fine-tuned, keeping the main visual encoder frozen.

### 3. Temporal Modeling and Fusion
- **Temporal Encoders:** The sequence of frame features from both the face and context streams are passed through separate Temporal Transformer models to capture temporal dynamics.
- **Fusion:** The resulting video-level embeddings for the face (`z_f`) and context (`z_c`) are then fused by concatenation and a linear projection to produce a final, unified visual embedding `z`.

### 4. Dual-View Prompting & MI Loss
To prevent the learnable prompts from overfitting and deviating from their intended semantics, a dual-view prompting strategy is used.
- **Hand-Crafted "Descriptive" View:** A rich, descriptive prompt for each class, detailing both behavioral cues and facial action unit (AU)-like micro-expressions (e.g., "A person with furrowed eyebrows and a puzzled gaze"). These are static.
- **Learnable "Soft" View:** CoOp-style learnable context vectors that are optimized during training.
- **Mutual Information (MI) Loss:** An InfoNCE-based loss is used to maximize the mutual information between the embeddings of the descriptive and soft prompts (`t_desc` and `t_soft`), ensuring the learnable prompts remain semantically grounded.

### 5. IEC (Instance-Enhanced Classifier)
To make the text-based classifier more adaptive to the visual features of each specific video instance, the IEC module is used.
- **Implementation:** Instead of a static text prototype for each class, a dynamic, instance-enhanced prototype is created using **Spherical Linear Interpolation (Slerp)**.
- **Formula:** `t_mix(k) = slerp(t_desc(k), z, λ_slerp)`, where `t_desc(k)` is the descriptive prompt for class `k`, `z` is the visual embedding for the video instance, and `λ_slerp` is a tunable weight.
- The final classification is performed by calculating the similarity between the visual embedding `z` and these mixed text prototypes `t_mix`.

### 6. Composite Loss Function
The model is trained with a composite loss:
`L_total = L_classification + λ_mi * L_mi`
- **`L_classification`**: A standard cross-entropy loss for the main classification task. A class-balanced weighting can be optionally applied to handle imbalanced datasets.
- **`L_mi`**: The Mutual Information loss to regularize the prompt learning.

## Weights Download
We provide the model weights trained by the method in this paper, which can be downloaded [here](https://drive.google.com/file/d/1mNYBKJ-vlsGf1QTN0tySs0-7sp-f7flb/view?usp=sharing).

**Important**: By downloading, accessing, or using the model weights, you agree to be bound by the terms and conditions of the license agreement located in the file weights/LICENSE-WEIGHTS.md, which permits use **only for non-commercial research purposes**. Please read the license carefully before downloading or using the weights.

If you do not agree to the non-commercial use terms of the license, please do not download or use the model weights.
## Performance
![image-20250702095507999](./imgs/performance.png)
## Visualizations

![image-20250702095507999](./imgs/vsualizations.png)

## Environment

The code is developed and tested under the following environment:

- Python 3.8

- PyTorch 2.2.2

- CUDA 12.4

```bash
conda create -n clip-caer python=3.8
conda activate clip-caer
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Training
The training process can be customized with several new command-line arguments to control the advanced features.

```bash
bash train.sh
```
For Google Colab users, a dedicated script is provided:
```bash
bash train_colab.sh
```

You can modify `train.sh` or `train_colab.sh` or pass arguments directly to `main.py`. The new arguments are:
- `--mi-loss-weight` (float, default: 0.5): Sets the weight for the Mutual Information loss.
- `--dc-loss-weight` (float, default: 0.1): Sets the weight for the Decorrelation loss.
- `--lr-adapter` (float, default: 1e-4): Defines the learning rate for the Expression-aware Adapter.
- `--slerp-weight` (float, default: 0.5): Controls the interpolation factor for the Instance-enhanced Classifier. Set to `0` to disable IEC.
- `--temperature` (float, default: 0.07): Temperature for the classification layer.
- `--class-balanced-loss`: (flag) Enable this to use class-balanced weights for the cross-entropy loss.
- `--logit-adj`: (flag) Enable this to use Logit Adjustment.
- `--logit-adj-tau` (float, default: 1.0): Temperature for Logit Adjustment.

Example of enabling all features:
```bash
python main.py \
    --mode train \
    --exper-name "EAA_IEC_MI_Balanced" \
    --gpu 0 \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.0003 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 0.001 \
    --lr-adapter 1e-4 \
    --mi-loss-weight 0.7 \
    --dc-loss-weight 1.2 \
    --slerp-weight 0.5 \
    --logit-adj \
    --logit-adj-tau 0.5 \
    --temperature 0.07 \
    --root-dir /kaggle/input/raer-video-emotion-dataset/RAER \
    --train-annotation /kaggle/input/raer-annot/annotation/train_80.txt \
    --test-annotation /kaggle/input/raer-annot/annotation/val_20.txt \
    --bounding-box-face /kaggle/input/raer-annot/annotation/bounding_box/face.json \
    --bounding-box-body /kaggle/input/raer-annot/annotation/bounding_box/body.json \
    --clip-path ViT-B/32 \
    --contexts-number 12 \
    ... # other arguments
```

### Evaluation
```bash
bash valid.sh
```
For Google Colab users, a dedicated script is provided:
```bash
bash valid_colab.sh
```

## Citations

If you find our paper useful in your research, please consider citing:

```bash
@InProceedings{Zhao_2025_ICCV,
    author    = {Zhao, Luming and Xuan, Jingwen and Lou, Jiamin and Yu, Yonghui and Yang, Wenwu},
    title     = {Context-Aware Academic Emotion Dataset and Benchmark},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025}
}
```

## Acknowledgment
Our codes are mainly based on [DFER-CLIP](https://github.com/zengqunzhao/DFER-CLIP/tree/main). Many thanks to the authors!