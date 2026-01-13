# main.py

# ==================== Imports ====================
import argparse
import datetime
import os
import random
import time
import warnings

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.Text import *
from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description="A highly configurable training script for RAER Dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group("Experiment & Environment", "Basic settings for the experiment")
exp_group.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
exp_group.add_argument("--eval-checkpoint", type=str, default="", help="Path to checkpoint for eval mode.")
exp_group.add_argument("--resume-from", type=str, default="", help="Path to checkpoint to resume training.")
exp_group.add_argument("--exper-name", type=str, default="test")
exp_group.add_argument("--dataset", type=str, default="RAER")
exp_group.add_argument("--gpu", type=str, default="0", help='GPU id, or "mps" or "cpu".')
exp_group.add_argument("--workers", type=int, default=4)
exp_group.add_argument("--seed", type=int, default=42)

# --- Data & Path ---
path_group = parser.add_argument_group("Data & Path", "Paths to datasets and pretrained models")
path_group.add_argument("--root-dir", type=str, default="./")
path_group.add_argument("--train-annotation", type=str, default="RAER/annotation/train.txt")
path_group.add_argument("--val-annotation", type=str, default="RAER/annotation/test.txt")
path_group.add_argument("--test-annotation", type=str, default="RAER/annotation/test.txt")
path_group.add_argument("--clip-path", type=str, default="ViT-B/32")
path_group.add_argument("--bounding-box-face", type=str, default="RAER/bounding_box/face.json")
path_group.add_argument("--bounding-box-body", type=str, default="RAER/bounding_box/body.json")

# --- Training Control ---
train_group = parser.add_argument_group("Training Control", "Parameters to control the training process")
train_group.add_argument("--epochs", type=int, default=50)
train_group.add_argument("--batch-size", type=int, default=8)
train_group.add_argument("--print-freq", type=int, default=10)
train_group.add_argument("--use-amp", action="store_true")
train_group.add_argument("--grad-clip", type=float, default=1.0)

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group("Optimizer & LR", "Hyperparameters for the optimizer and scheduler")
optim_group.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "AdamW"])
optim_group.add_argument("--lr", type=float, default=1e-4)
optim_group.add_argument("--lr-image-encoder", type=float, default=0.0)
optim_group.add_argument("--lr-prompt-learner", type=float, default=1e-4)
optim_group.add_argument("--lr-adapter", type=float, default=1e-4)
optim_group.add_argument("--weight-decay", type=float, default=1e-4)
optim_group.add_argument("--momentum", type=float, default=0.9)
optim_group.add_argument("--milestones", nargs="+", type=int, default=[20, 35])
optim_group.add_argument("--gamma", type=float, default=0.1)

# --- Loss & Imbalance Handling ---
loss_group = parser.add_argument_group("Loss & Imbalance Handling", "Loss functions and imbalance handling")
loss_group.add_argument("--lambda_mi", type=float, default=0.0)
loss_group.add_argument("--lambda_dc", type=float, default=0.0)
loss_group.add_argument("--mi-warmup", type=int, default=0)
loss_group.add_argument("--mi-ramp", type=int, default=0)
loss_group.add_argument("--dc-warmup", type=int, default=0)
loss_group.add_argument("--dc-ramp", type=int, default=0)

loss_group.add_argument("--class-balanced-loss", action="store_true", help="Use FocalLoss.")
loss_group.add_argument("--use-weighted-sampler", action="store_true", help="Use WeightedRandomSampler.")
loss_group.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor (LSR2).")

loss_group.add_argument("--logit-adj", action="store_true", help="Use logit adjustment (train priors).")
loss_group.add_argument("--logit-adj-tau", type=float, default=0.5, help="Tau for logit adjustment.")

# --- 2-Head Loss & Inference ---
loss_group.add_argument("--two-head-loss", action="store_true", help="Enable 2-head loss for Confusion splitting.")
loss_group.add_argument("--w-bin", type=float, default=0.5, help="Weight for the binary auxiliary loss.")
loss_group.add_argument("--w-4", type=float, default=0.5, help="Weight for the 4-class auxiliary loss.")
loss_group.add_argument("--sweep-range", type=float, nargs=3, default=[0.2, 0.81, 0.05], help="Sweep range for soft-gate threshold (start, stop, step).")
loss_group.add_argument("--conf-recall-min", type=float, default=0.0, help="Minimum confusion recall to consider a threshold as best.")

exp_group.add_argument("--reset-epoch", action="store_true",
                       help="When resuming, ignore checkpoint epoch and start from 0 (recommended for stage2).")
# --- Model & Input ---
model_group = parser.add_argument_group("Model & Input", "Model architecture and input handling")
model_group.add_argument(
    "--text-type",
    default="class_descriptor",
    choices=["class_names", "class_names_with_context", "class_descriptor",
             "class_descriptor_only_face", "class_descriptor_only_body", "prompt_ensemble"],
)
model_group.add_argument("--temporal-layers", type=int, default=1)
model_group.add_argument("--contexts-number", type=int, default=12)
model_group.add_argument("--class-token-position", type=str, default="end")
model_group.add_argument("--class-specific-contexts", type=str, default="True", choices=["True", "False"])
model_group.add_argument("--load_and_tune_prompt_learner", type=str, default="True", choices=["True", "False"])
model_group.add_argument("--num-segments", type=int, default=16)
model_group.add_argument("--duration", type=int, default=1)
model_group.add_argument("--image-size", type=int, default=224)
model_group.add_argument("--slerp-weight", type=float, default=0.0)
model_group.add_argument("--temperature", type=float, default=0.07)
model_group.add_argument("--crop-body", action="store_true", help="Crop body from the input images.")

# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    # device
    if args.gpu == "mps":
        args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    elif args.gpu == "cpu":
        args.device = torch.device("cpu")
    else:
        if torch.cuda.is_available() and str(args.gpu).isdigit():
            args.device = torch.device(f"cuda:{args.gpu}")
        else:
            args.device = torch.device("cpu")

    print(f"Using device: {args.device}")

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H-%M]")  # avoid ":" for zip on windows
    args.name = args.exper_name + time_str
    args.output_path = os.path.join("outputs", args.name)
    os.makedirs(args.output_path, exist_ok=True)

    print("************************")
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f"{k} = {v}")
    print("************************")

    log_txt_path = os.path.join(args.output_path, "log.txt")
    with open(log_txt_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {v}\n")
        f.write("*" * 50 + "\n\n")

    return args


def maybe_resume_training(args, model, optimizer, scheduler):
    """
    Resume model/optimizer/scheduler + set start_epoch from checkpoint.
    Expect checkpoint saved by your save_checkpoint with keys:
    - epoch
    - state_dict
    - optimizer
    """
    start_epoch = 0
    best_val_uar = 0.0

    if not args.resume_from:
        return start_epoch, best_val_uar

    ckpt_path = args.resume_from
    print(f"=> Resuming training from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    # model
    if "state_dict" in ckpt:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        # allow raw state_dict
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")


    # optimizer/scheduler (optional but recommended)
    if isinstance(ckpt, dict) and "optimizer" in ckpt and optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("=> Optimizer state loaded.")
        except Exception as e:
            print(f"[WARN] Cannot load optimizer state: {e}")

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        start_epoch = int(ckpt["epoch"])
        print(f"=> start_epoch set to {start_epoch}")

    if isinstance(ckpt, dict) and "best_acc" in ckpt:
        try:
            best_val_uar = float(ckpt["best_acc"])
        except Exception:
            pass
    
    if args.reset_epoch:
        print(f"=> [--reset-epoch] was set. Resetting start_epoch to 0 from {start_epoch}.")
        start_epoch = 0

    return start_epoch, best_val_uar


# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    log_txt_path = os.path.join(args.output_path, "log.txt")
    log_curve_path = os.path.join(args.output_path, "log.png")
    log_confusion_matrix_path = os.path.join(args.output_path, "confusion_matrix.png")
    checkpoint_path = os.path.join(args.output_path, "model.pth")
    best_checkpoint_path = os.path.join(args.output_path, "model_best.pth")

    # Build model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)  # from utils/builders + models/Text
    model = build_model(args, input_text).to(args.device)
    print("=> Model built.")

    # Load data
    print("=> Building dataloaders...")
    # Your builders.py should return (train_loader, val_loader, test_loader)
    train_loader, val_loader, test_loader = build_dataloaders(args)
    print("=> Dataloaders built.")

    # Print validation set stats
    val_class_counts = get_class_counts(args.val_annotation)
    print(f"=> Validation set class counts: {val_class_counts}")


    # ---------------- Loss ----------------
    # class_counts must be from TRAIN annotation
    class_counts = get_class_counts(args.train_annotation)  # your utils.utils should implement this

    if args.label_smoothing > 0 and not args.two_head_loss:
        criterion = LSR2(e=args.label_smoothing, label_mode="raer").to(args.device)
    elif args.class_balanced_loss and not args.two_head_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2).to(args.device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

    mi_criterion = MILoss().to(args.device) if args.lambda_mi > 0 else None
    dc_criterion = DCLoss().to(args.device) if args.lambda_dc > 0 else None

    class_priors = None
    if args.logit_adj:
        class_priors = torch.tensor(class_counts, dtype=torch.float32)
        class_priors = class_priors / (class_priors.sum() + 1e-12)
        # keep on cpu is fine; trainer moves to output.device
        print(f"=> Logit adjustment ON. priors={class_priors.tolist()}, tau={args.logit_adj_tau}")

    # ---------------- Optimizer ----------------
    optimizer_grouped_parameters = [
        {"params": model.temporal_net.parameters(), "lr": args.lr},
        {"params": model.temporal_net_body.parameters(), "lr": args.lr},
        {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
        {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
        {"params": model.project_fc.parameters(), "lr": args.lr},
        {"params": model.face_adapter.parameters(), "lr": args.lr_adapter},
    ]
    if hasattr(model, 'cls_bin'):
        optimizer_grouped_parameters.append({"params": model.cls_bin.parameters(), "lr": args.lr})
    if hasattr(model, 'cls_4'):
        optimizer_grouped_parameters.append({"params": model.cls_4.parameters(), "lr": args.lr})

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    trainer = Trainer(
        model, criterion, optimizer, scheduler, args.device, log_txt_path,
        mi_criterion=mi_criterion, lambda_mi=args.lambda_mi,
        dc_criterion=dc_criterion, lambda_dc=args.lambda_dc,
        class_priors=class_priors, logit_adj_tau=args.logit_adj_tau,
        mi_warmup=args.mi_warmup, mi_ramp=args.mi_ramp,
        dc_warmup=args.dc_warmup, dc_ramp=args.dc_ramp,
        use_amp=args.use_amp, grad_clip=args.grad_clip,
        two_head_loss=args.two_head_loss, w_bin=args.w_bin, w_4=args.w_4,
        sweep_range=args.sweep_range, conf_recall_min=args.conf_recall_min
    )

    # Resume if requested
    start_epoch, best_val_uar = maybe_resume_training(args, model, optimizer, scheduler)

    recorder = RecorderMeter(args.epochs)
    best_val_war = 0.0
    best_train_uar = 0.0
    best_train_war = 0.0

    for epoch in range(start_epoch, args.epochs):
        inf = f"******************** Epoch: {epoch} ********************"
        start_time = time.time()
        print(inf)
        with open(log_txt_path, "a") as f:
            f.write(inf + "\n")

        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = " ".join([f"{lr:.1e}" for lr in current_lrs])
        log_msg = f"Current learning rates: {lr_str}"
        print(log_msg)
        with open(log_txt_path, "a") as f:
            f.write(log_msg + "\n")

        train_war, train_uar, train_los, train_cm = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, val_cm = trainer.validate(val_loader, str(epoch))
        scheduler.step()

        is_best = val_uar > best_val_uar
        best_val_uar = max(val_uar, best_val_uar)
        best_val_war = max(val_war, best_val_war)
        best_train_uar = max(train_uar, best_train_uar)
        best_train_war = max(train_war, best_train_war)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_val_uar,
                "optimizer": optimizer.state_dict(),
                "recorder": recorder,
            },
            is_best,
            checkpoint_path,
            best_checkpoint_path,
        )

        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_war, train_uar, val_los, val_war, val_uar)
        recorder.plot_curve(log_curve_path)

        log_msg = (
            f"\n--- Epoch {epoch} Summary ---
"
            f"Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n"
            f"Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n"
            f"Best Valid UAR so far: {best_val_uar:.2f}%\n"
            f"Time: {epoch_time:.2f}s\n"
            f"Train Confusion Matrix:\n{train_cm}\n"
            f"Validation Confusion Matrix:\n{val_cm}\n"
            f"--- End of Epoch {epoch} ---
"
        )
        print(log_msg)
        with open(log_txt_path, "a") as f:
            f.write(log_msg + "\n\n")

    # Final evaluation with best model
    print("=> Final evaluation on test set...")
    ckpt = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    computer_uar_war(
        val_loader=test_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset} Test Set",
    )


def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, "log.txt")
    log_confusion_matrix_path = os.path.join(args.output_path, "confusion_matrix.png")

    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text).to(args.device)

    if not args.eval_checkpoint:
        raise ValueError("--eval-checkpoint is required in eval mode.")

    ckpt = torch.load(args.eval_checkpoint, map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    _, _, test_loader = build_dataloaders(args)

    computer_uar_war(
        val_loader=test_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}",
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == "__main__":
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)

    if args.mode == "eval":
        run_eval(args)
    else:
        run_training(args)
