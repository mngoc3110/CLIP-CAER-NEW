# ==================== Imports ====================
import argparse
import datetime
import os
import random
import shutil
import time

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import warnings
from clip import clip
from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *

# Ignore specific warnings (for cleaner output)
warnings.filterwarnings("ignore", category=UserWarning)
# Use 'Agg' backend for matplotlib (no GUI required)
matplotlib.use('Agg')

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description='A highly configurable training script for RAER Dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group('Experiment & Environment', 'Basic settings for the experiment')
exp_group.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help="Execution mode: 'train' for a full training run, 'eval' for evaluation only.")
exp_group.add_argument('--eval-checkpoint', type=str,
                       help="Path to the model checkpoint for evaluation mode (e.g., outputs/exp_name/model_best.pth).")
exp_group.add_argument('--exper-name', type=str, default='test', help='A name for the experiment to create a unique output folder.')
exp_group.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
exp_group.add_argument('--gpu', type=int, default=2, help='ID of the GPU to use.')
exp_group.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
exp_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

# --- Data & Path ---
path_group = parser.add_argument_group('Data & Path', 'Paths to datasets and pretrained models')
path_group.add_argument('--root-dir', type=str, help='Root directory of the dataset. E.g., /kaggle/input/raer-video-emotion-dataset/RAER')
path_group.add_argument('--train-annotation', type=str, help='Absolute path to training annotation file. E.g., /kaggle/input/raer-annot/annotation/train_abs.txt')
path_group.add_argument('--val-annotation', type=str, help='Absolute path to validation annotation file. E.g., /kaggle/input/raer-annot/annotation/val_20.txt')
path_group.add_argument('--test-annotation', type=str, help='Absolute path to testing annotation file. E.g., /kaggle/input/raer-annot/annotation/test_abs.txt')
path_group.add_argument('--clip-path', type=str, help='Path to the pretrained CLIP model.')
path_group.add_argument('--bounding-box-face', type=str, help='Absolute path to face bounding box JSON. E.g., /kaggle/input/raer-annot/annotation/bounding_box/face_abs.json')
path_group.add_argument('--bounding-box-body', type=str, help='Absolute path to body bounding box JSON. E.g., /kaggle/input/raer-annot/annotation/bounding_box/body_abs.json')

# --- Training Control ---
train_group = parser.add_argument_group('Training Control', 'Parameters to control the training process')
train_group.add_argument('--epochs', type=int, default=50, help='Total number of training epochs.')
train_group.add_argument('--batch-size', type=int, default=8, help='Batch size for training and validation.')
train_group.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training logs.')
train_group.add_argument('--use-amp', action='store_true', help='Use Automatic Mixed Precision.')
train_group.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value.')

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group('Optimizer & LR', 'Hyperparameters for the optimizer and scheduler')
optim_group.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate for main modules.')
optim_group.add_argument('--lr-image-encoder', type=float, default=1e-6, help='Learning rate for the image encoder part.')
optim_group.add_argument('--lr-prompt-learner', type=float, default=0.001, help='Learning rate for the prompt learner.')
optim_group.add_argument('--lr-adapter', type=float, default=1e-5, help='Learning rate for the adapter.')
optim_group.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay for the optimizer.')
optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
optim_group.add_argument('--milestones', nargs='+', type=int, default=[20, 35], help='Epochs at which to decay the learning rate.')
optim_group.add_argument('--gamma', type=float, default=0.1, help='Factor for learning rate decay.')

# --- Loss & Imbalance Handling ---
loss_group = parser.add_argument_group('Loss & Imbalance Handling', 'Parameters for loss functions and imbalance handling')
loss_group.add_argument('--lambda_mi', type=float, default=0.7, help='Weight for the Mutual Information loss.')
loss_group.add_argument('--lambda_dc', type=float, default=1.2, help='Weight for the Decorrelation loss.')
loss_group.add_argument('--mi-warmup', type=int, default=3, help='Warmup epochs for MI loss.')
loss_group.add_argument('--mi-ramp', type=int, default=8, help='Ramp-up epochs for MI loss.')
loss_group.add_argument('--dc-warmup', type=int, default=5, help='Warmup epochs for DC loss.')
loss_group.add_argument('--dc-ramp', type=int, default=10, help='Ramp-up epochs for DC loss.')
loss_group.add_argument('--class-balanced-loss', action='store_true', help='Use class-balanced loss.')
loss_group.add_argument('--logit-adj', action='store_true', help='Use logit adjustment.')
loss_group.add_argument('--logit-adj-tau', type=float, default=0.5, help='Temperature for logit adjustment.')
loss_group.add_argument('--use-weighted-sampler', action='store_true', help='Use WeightedRandomSampler.')
loss_group.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing factor.')

# --- Model & Input ---
model_group = parser.add_argument_group('Model & Input', 'Parameters for model architecture and data handling')
model_group.add_argument('--text-type', default='class_descriptor', choices=['class_names', 'class_names_with_context', 'class_descriptor'], help='Type of text prompts to use.')
model_group.add_argument('--temporal-layers', type=int, default=1, help='Number of layers in the temporal modeling part.')
model_group.add_argument('--contexts-number', type=int, default=12, help='Number of context vectors in the prompt learner.')
model_group.add_argument('--class-token-position', type=str, default="end", help='Position of the class token in the prompt.')
model_group.add_argument('--class-specific-contexts', type=str, default='True', choices=['True', 'False'], help='Whether to use class-specific context prompts.')
model_group.add_argument('--load_and_tune_prompt_learner', type=str, default='True', choices=['True', 'False'], help='Whether to load and fine-tune the prompt learner.')
model_group.add_argument('--num-segments', type=int, default=16, help='Number of segments to sample from each video.')
model_group.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
model_group.add_argument('--image-size', type=int, default=224, help='Size to resize input images to.')
model_group.add_argument('--slerp-weight', type=float, default=0.5, help='Weight for spherical linear interpolation (IEC).')
model_group.add_argument('--temperature', type=float, default=0.07, help='Temperature for the classification layer.')
model_group.add_argument('--crop-body', action='store_true', help='Crop body from the input images.')

# --- Staged Training ---
staged_group = parser.add_argument_group('Staged Training', 'Parameters for staged training')
staged_group.add_argument('--staged-training', action='store_true', help='Enable staged training.')
staged_group.add_argument('--epochs-stage1', type=int, default=10, help='Number of epochs for stage 1 (Prompt Learner).')
staged_group.add_argument('--epochs-stage2', type=int, default=20, help='Number of epochs for stage 2 (Adapter & Temporal).')
staged_group.add_argument('--epochs-stage3', type=int, default=20, help='Number of epochs for stage 3 (End-to-End).')
staged_group.add_argument('--lr-stage1', type=float, default=0.001, help='Learning rate for prompt learner in stage 1.')
staged_group.add_argument('--lr-stage2', type=float, default=1e-4, help='Learning rate for adapter/temporal in stage 2.')
staged_group.add_argument('--lr-stage3', type=float, default=1e-5, help='Learning rate for end-to-end fine-tuning in stage 3.')


# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    
    print("Environment and random seeds set successfully.")
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H:%M]")
    
    args.name = args.exper_name + time_str
        
    args.output_path = os.path.join("outputs", args.name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('************************')
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('************************')
    
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    with open(log_txt_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k} = {v}\n')
        f.write('*'*50 + '\n\n')
        
    return args

def set_trainable_params(model, stage):
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    if stage == 1:
        print("Setting trainable parameters for STAGE 1: Prompt Learner")
        trainable_keywords = ["prompt_learner"]
    elif stage == 2:
        print("Setting trainable parameters for STAGE 2: Adapter & Temporal")
        trainable_keywords = ["face_adapter", "temporal_net", "temporal_net_body", "project_fc"]
    elif stage == 3:
        print("Setting trainable parameters for STAGE 3: End-to-End Fine-tuning")
        trainable_keywords = ["image_encoder", "temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "face_adapter"]
    else: # Default (non-staged) training
        print("Setting trainable parameters for default training")
        trainable_keywords = ["image_encoder", "temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "face_adapter"]

    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')


# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    # Paths for logging and saving
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_curve_path = os.path.join(args.output_path, 'log.png')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    checkpoint_path = os.path.join(args.output_path, 'model.pth')
    best_checkpoint_path = os.path.join(args.output_path, 'model_best.pth')        
    best_train_uar = 0.0
    best_train_war = 0.0
    best_val_uar = 0.0
    best_val_war = 0.0
    start_epoch = 0
    
    
    # Build model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)
    print("=> Model built and moved to device successfully.")

    # Load data
    print("=> Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(args)
    print("=> Dataloaders built successfully.")

    # Loss and optimizer
    class_counts = get_class_counts(args.train_annotation)
    
    if args.label_smoothing > 0:
        criterion = LSR2(e=args.label_smoothing, label_mode='class_descriptor').to(args.device)
    elif args.class_balanced_loss:
        print("=> Using class-balanced loss.")
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device)).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)

    mi_criterion = MILoss().to(args.device) if args.lambda_mi > 0 else None
    dc_criterion = DCLoss().to(args.device) if args.lambda_dc > 0 else None

    class_priors = None
    if args.logit_adj:
        print("=> Using logit adjustment.")
        class_priors = torch.tensor(class_counts, dtype=torch.float) / sum(class_counts)
        class_priors = class_priors.to(args.device)

    # --- Staged Training Loop ---
    if args.staged_training:
        total_epochs = args.epochs_stage1 + args.epochs_stage2 + args.epochs_stage3
        recorder = RecorderMeter(total_epochs)
        
        # STAGE 1: Train Prompt Learner
        print("\n\n" + "="*40)
        print(" " * 15 + "STAGE 1")
        print("="*40 + "\n")
        set_trainable_params(model, 1)
        optimizer = torch.optim.SGD([{'params': model.prompt_learner.parameters(), 'lr': args.lr_stage1}],
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        trainer = Trainer(model, criterion, optimizer, scheduler, args.device, log_txt_path, 
                        mi_criterion=mi_criterion, lambda_mi=args.lambda_mi, dc_criterion=dc_criterion, lambda_dc=args.lambda_dc,
                        class_priors=class_priors, logit_adj_tau=args.logit_adj_tau,
                        mi_warmup=args.mi_warmup, mi_ramp=args.mi_ramp, dc_warmup=args.dc_warmup, dc_ramp=args.dc_ramp, use_amp=args.use_amp, grad_clip=args.grad_clip)
        
        for epoch in range(args.epochs_stage1):
            run_one_training_epoch(trainer, train_loader, val_loader, checkpoint_path, best_checkpoint_path, epoch, recorder, best_val_uar, best_val_war, best_train_uar, best_train_war, log_txt_path, log_curve_path)

        # STAGE 2: Train Adapter and Temporal Models
        print("\n\n" + "="*40)
        print(" " * 15 + "STAGE 2")
        print("="*40 + "\n")
        set_trainable_params(model, 2)
        optimizer = torch.optim.SGD([
            {"params": model.face_adapter.parameters(), "lr": args.lr_stage2},
            {"params": model.temporal_net.parameters(), "lr": args.lr_stage2},
            {"params": model.temporal_net_body.parameters(), "lr": args.lr_stage2},
            {"params": model.project_fc.parameters(), "lr": args.lr_stage2},
        ], momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        for epoch in range(args.epochs_stage1, args.epochs_stage1 + args.epochs_stage2):
             run_one_training_epoch(trainer, train_loader, val_loader, checkpoint_path, best_checkpoint_path, epoch, recorder, best_val_uar, best_val_war, best_train_uar, best_train_war, log_txt_path, log_curve_path)

        # STAGE 3: End-to-End Fine-tuning
        print("\n\n" + "="*40)
        print(" " * 15 + "STAGE 3")
        print("="*40 + "\n")
        set_trainable_params(model, 3)
        optimizer = torch.optim.SGD([
            {"params": model.parameters(), "lr": args.lr_stage3}
        ], momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        for epoch in range(args.epochs_stage1 + args.epochs_stage2, total_epochs):
             run_one_training_epoch(trainer, train_loader, val_loader, checkpoint_path, best_checkpoint_path, epoch, recorder, best_val_uar, best_val_war, best_train_uar, best_train_war, log_txt_path, log_curve_path)

    else: # Default training
        recorder = RecorderMeter(args.epochs)
        optimizer = torch.optim.SGD([
            {"params": model.temporal_net.parameters(), "lr": args.lr},
            {"params": model.temporal_net_body.parameters(), "lr": args.lr},
            {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
            {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
            {"params": model.project_fc.parameters(), "lr": args.lr_image_encoder},
            {"params": model.face_adapter.parameters(), "lr": args.lr_adapter}
        ], momentum=args.momentum, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        trainer = Trainer(model, criterion, optimizer, scheduler, args.device, log_txt_path, 
                        mi_criterion=mi_criterion, lambda_mi=args.lambda_mi,
                        dc_criterion=dc_criterion, lambda_dc=args.lambda_dc,
                        class_priors=class_priors, logit_adj_tau=args.logit_adj_tau,
                        mi_warmup=args.mi_warmup, mi_ramp=args.mi_ramp,
                        dc_warmup=args.dc_warmup, dc_ramp=args.dc_ramp, use_amp=args.use_amp, grad_clip=args.grad_clip)
        for epoch in range(start_epoch, args.epochs):
             run_one_training_epoch(trainer, train_loader, val_loader, checkpoint_path, best_checkpoint_path, epoch, recorder, best_val_uar, best_val_war, best_train_uar, best_train_war, log_txt_path, log_curve_path)

    # Final evaluation with best model
    print("=> Final evaluation on test set...")
    pre_trained_dict = torch.load(best_checkpoint_path,map_location=f"cuda:{args.gpu}")['state_dict']
    model.load_state_dict(pre_trained_dict)
    computer_uar_war(
        val_loader=test_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset} Test Set"
    )

def run_one_training_epoch(trainer, train_loader, val_loader, checkpoint_path, best_checkpoint_path, epoch, recorder, best_val_uar, best_val_war, best_train_uar, best_train_war, log_txt_path, log_curve_path):
    inf = f'******************** Epoch: {epoch} ********************'
    start_time = time.time()
    print(inf)
    with open(log_txt_path, 'a') as f:
        f.write(inf + '\n')

    # Log current learning rates
    current_lrs = [param_group['lr'] for param_group in trainer.optimizer.param_groups]
    lr_str = ' '.join([f'{lr:.1e}' for lr in current_lrs])
    log_msg = f'Current learning rates: {lr_str}'
    with open(log_txt_path, 'a') as f:
        f.write(log_msg + '\n')
    print(log_msg)

    # Train & Validate
    train_war, train_uar, train_los, _ = trainer.train_epoch(train_loader, epoch)
    val_war, val_uar, val_los, val_cm = trainer.validate(val_loader, str(epoch))
    trainer.scheduler.step()

    # Save checkpoint
    is_best = val_uar > best_val_uar
    best_val_uar = max(val_uar, best_val_uar)
    best_val_war = max(val_war, best_val_war)
    best_train_uar = max(train_uar, best_train_uar)
    best_train_war = max(train_war, best_train_war)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': trainer.model.state_dict(),
        'best_acc': best_val_uar, 
        'optimizer': trainer.optimizer.state_dict(),
        'recorder': recorder
    }, is_best, checkpoint_path, best_checkpoint_path)

    # Record metrics
    epoch_time = time.time() - start_time
    recorder.update(epoch, train_los, train_war, train_uar, val_los, val_war, val_uar)
    recorder.plot_curve(log_curve_path)
    
    log_msg = (
               f'Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n'
               f'Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n'
               f'Best Valid WAR: {best_val_war:.2f}% | Best Valid UAR: {best_val_uar:.2f}%\n'
               f'Best Train WAR: {best_train_war:.2f}% | Best Train UAR: {best_train_uar:.2f}%\n'
               f'An epoch time: {epoch_time:.2f}s\n'
               f'Validation Confusion Matrix:\n{val_cm}')
    print(log_msg)
    with open(log_txt_path, 'a') as f:
        f.write(log_msg + '\n\n')

def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')

    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # Load pretrained weights
    model.load_state_dict(torch.load(args.eval_checkpoint,map_location=args.device)['state_dict'])

    # Load data
    _, _, test_loader = build_dataloaders(args)

    # Run evaluation
    computer_uar_war(
        val_loader=test_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}"
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == '__main__':
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)
    
    if args.mode == 'eval':
        run_eval(args)
    else:
        run_training(args)