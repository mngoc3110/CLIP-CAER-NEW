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

from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description="RAER Training – Stage 2 (Confusion Split)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# ---------- Experiment ----------
parser.add_argument("--mode", default="train", choices=["train", "eval"])
parser.add_argument("--eval-checkpoint", default="")
parser.add_argument("--resume-from", default="")
parser.add_argument("--exper-name", default="stage2_conf_split")
parser.add_argument("--dataset", default="RAER")
parser.add_argument("--gpu", default="0")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)

# ---------- Paths ----------
parser.add_argument("--root-dir", required=True)
parser.add_argument("--train-annotation", required=True)
parser.add_argument("--val-annotation", required=True)
parser.add_argument("--test-annotation", required=True)
parser.add_argument("--clip-path", default="ViT-B/32")
parser.add_argument("--bounding-box-face", required=True)
parser.add_argument("--bounding-box-body", required=True)

# ---------- Training ----------
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--print-freq", type=int, default=10)
parser.add_argument("--use-amp", action="store_true")
parser.add_argument("--grad-clip", type=float, default=1.0)

# ---------- Optimizer ----------
parser.add_argument("--optimizer", default="AdamW", choices=["SGD", "AdamW"])
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr-image-encoder", type=float, default=1e-6)
parser.add_argument("--lr-prompt-learner", type=float, default=5e-4)
parser.add_argument("--lr-adapter", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--milestones", nargs="+", type=int, default=[20, 35])
parser.add_argument("--gamma", type=float, default=0.1)

# ---------- Loss ----------
parser.add_argument("--lambda-mi", type=float, default=0.05)
parser.add_argument("--lambda-dc", type=float, default=0.08)
parser.add_argument("--mi-warmup", type=int, default=5)
parser.add_argument("--mi-ramp", type=int, default=10)
parser.add_argument("--dc-warmup", type=int, default=5)
parser.add_argument("--dc-ramp", type=int, default=10)

parser.add_argument("--use-weighted-sampler", action="store_true")

# ---------- Stage-2: 2-Head ----------
parser.add_argument("--two-head-loss", action="store_true")
parser.add_argument("--w-bin", type=float, default=1.0)
parser.add_argument("--w-4", type=float, default=1.0)
parser.add_argument("--soft-gate-thr", type=float, default=0.65)

# ---------- Model ----------
parser.add_argument("--text-type", default="prompt_ensemble")
parser.add_argument("--contexts-number", type=int, default=12)
parser.add_argument("--class-token-position", default="end")
parser.add_argument("--class-specific-contexts", default="True")
parser.add_argument("--load_and_tune_prompt_learner", default="True")
parser.add_argument("--temporal-layers", type=int, default=1)
parser.add_argument("--num-segments", type=int, default=16)
parser.add_argument("--duration", type=int, default=1)
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--crop-body", action="store_true")

# ==================== Helpers ====================
def setup_env(args):
    if args.gpu == "cpu":
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True
    return args


def setup_output(args):
    ts = datetime.datetime.now().strftime("[%m-%d]-[%H-%M]")
    args.name = f"{args.exper_name}-{ts}"
    args.output_path = os.path.join("outputs", args.name)
    os.makedirs(args.output_path, exist_ok=True)

    log_path = os.path.join(args.output_path, "log.txt")
    with open(log_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {v}\n")
        f.write("=" * 50 + "\n")
    return log_path


# ==================== Training ====================
def run_training(args):
    log_txt_path = setup_output(args)

    # ---- Model ----
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text).to(args.device)

    # ---- Data ----
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # ---- Loss ----
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    mi_criterion = MILoss().to(args.device)
    dc_criterion = DCLoss().to(args.device)

    # ---- Optimizer ----
    params = [
        {"params": model.temporal_net.parameters(), "lr": args.lr},
        {"params": model.temporal_net_body.parameters(), "lr": args.lr},
        {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
        {"params": model.face_adapter.parameters(), "lr": args.lr_adapter},
        {"params": model.project_fc.parameters(), "lr": args.lr},
        {"params": model.cls_bin.parameters(), "lr": args.lr},
        {"params": model.cls_4.parameters(), "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        log_txt_path=log_txt_path,
        mi_criterion=mi_criterion,
        lambda_mi=args.lambda_mi,
        dc_criterion=dc_criterion,
        lambda_dc=args.lambda_dc,
        mi_warmup=args.mi_warmup,
        mi_ramp=args.mi_ramp,
        dc_warmup=args.dc_warmup,
        dc_ramp=args.dc_ramp,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        two_head_loss=True,
        w_bin=args.w_bin,
        w_4=args.w_4,
        soft_gate_thr=args.soft_gate_thr,
    )

    # ---- Resume ----
    start_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)

    best_uar = 0.0
    best_ckpt = os.path.join(args.output_path, "model_best.pth")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch} =====")
        train_war, train_uar, _, _ = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, _, val_cm = trainer.validate(val_loader, epoch)

        scheduler.step()

        if val_uar > best_uar:
            best_uar = val_uar
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_uar,
                },
                best_ckpt,
            )

        print(f"Epoch {epoch} | Train UAR {train_uar:.2f} | Val UAR {val_uar:.2f}")

    print("Best Val UAR:", best_uar)

    # ---- Final Test ----
    ckpt = torch.load(best_ckpt, map_location=args.device)
    model.load_state_dict(ckpt["state_dict"])
    computer_uar_war(
        test_loader, model, args.device, class_names,
        log_txt_path=log_txt_path,
        title="Final Test Confusion Matrix"
    )


# ==================== Entry ====================
if __name__ == "__main__":
    args = parser.parse_args()
    args = setup_env(args)
    run_training(args)