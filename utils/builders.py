# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')

    print("\nInput Text Prompts:")
    if any(isinstance(i, list) for i in input_text):
        for i, class_prompts in enumerate(input_text):
            print(f"- Class {i}: {class_prompts}")
    else:
        for text in input_text:
            print(text)

    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.lr_image_encoder > 0:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

    trainable_params_keywords = ["temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "face_adapter"]
    
    if hasattr(model, 'cls_bin'):
        trainable_params_keywords.append('cls_bin')
    if hasattr(model, 'cls_4'):
        trainable_params_keywords.append('cls_4')

    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
        class_descriptor = class_descriptor_5
        ensemble_prompts = prompt_ensemble_5
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    elif args.text_type == "prompt_ensemble":
        input_text = ensemble_prompts
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text


def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = args.train_annotation
    val_annotation_file_path = args.val_annotation
    test_annotation_file_path = args.test_annotation
    
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    print("Loading train data...")
    train_data = train_data_loader(
        root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )
    print(f"Total number of training images: {len(train_data)}")
    
    print("Loading validation data...")
    val_data = test_data_loader(
        root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )

    print("Loading test data...")
    test_data = test_data_loader(
        root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )

    print("Creating DataLoader instances...")
    
    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        print("=> Using WeightedRandomSampler.")
        class_counts = get_class_counts(train_annotation_file_path)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        
        sample_weights = []
        with open(train_annotation_file_path, 'r') as f:
            for line in f:
                try:
                    label = int(line.strip().split()[2]) - 1 
                    if 0 <= label < len(class_weights):
                        sample_weights.append(class_weights[label])
                    else:
                        print(f"Warning: Found invalid label '{label+1}' in {train_annotation_file_path}")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse line: {line.strip()}")
        
        if len(sample_weights) == len(train_data):
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False
        else:
            print(f"Warning: Mismatch between number of samples ({len(train_data)}) and weights ({len(sample_weights)}). Disabling sampler.")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader