import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import confusion_matrix

from utils.utils import AverageMeter, ProgressMeter, get_loss_weight


class Trainer:
    """
    Trainer with:
    - Stage-2 2-head loss (binary + 4-class)
    - Optional MI / DC
    - Correct UAR computation (5-class)
    """

    CONFUSION_ID = 2  # RAER label

    def __init__(
        self,
        model,
        criterion_bin,     # Binary loss (CE or Focal)
        criterion4,        # 4-class loss (LSR2 or CE)
        optimizer,
        scheduler,
        device,
        log_txt_path,
        mi_criterion=None,
        lambda_mi=0.0,
        dc_criterion=None,
        lambda_dc=0.0,
        w_bin=1.0,
        w_4=1.0,
        soft_gate_thr=0.7,
        use_amp=False,
        grad_clip=1.0,
    ):
        self.model = model
        self.criterion_bin = criterion_bin
        self.criterion4 = criterion4
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_txt_path = log_txt_path

        self.mi_criterion = mi_criterion
        self.lambda_mi = lambda_mi
        self.dc_criterion = dc_criterion
        self.lambda_dc = lambda_dc

        self.w_bin = w_bin
        self.w_4 = w_4
        self.soft_gate_thr = soft_gate_thr

        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.print_freq = 10
        os.makedirs("debug_predictions", exist_ok=True)

    def _reduce_ensemble(self, text_feat):
        if hasattr(self.model, "is_ensemble") and self.model.is_ensemble:
            C, P = self.model.num_classes, self.model.num_prompts_per_class
            return text_feat.view(C, P, -1).mean(dim=1)
        return text_feat

    def _run_one_epoch(self, loader, epoch, train=True):
        self.model.train() if train else self.model.eval()
        prefix = f"{'Train' if train else 'Valid'} Epoch [{epoch}]"

        loss_meter = AverageMeter("Loss", ":.4e")
        war_meter = AverageMeter("WAR", ":6.2f")

        all_preds, all_targets = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i, (img_f, img_b, target) in enumerate(loader):
                img_f = img_f.to(self.device)
                img_b = img_b.to(self.device)
                target = target.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out5, log_bin, log4, txt_feat, hc_feat = self.model(img_f, img_b)

                    # ===== Binary =====
                    tgt_bin = (target == self.CONFUSION_ID).long()
                    loss_bin = self.criterion_bin(log_bin, tgt_bin)

                    # ===== 4-class =====
                    mask = target != self.CONFUSION_ID
                    if mask.sum() > 0:
                        tgt4 = target[mask]
                        tgt4 = torch.where(tgt4 > self.CONFUSION_ID, tgt4 - 1, tgt4)
                        loss4 = self.criterion4(log4[mask], tgt4)
                    else:
                        loss4 = torch.tensor(0.0, device=self.device)

                    loss = self.w_bin * loss_bin + self.w_4 * loss4

                    # ===== MI / DC =====
                    if train and self.mi_criterion:
                        txt_red = self._reduce_ensemble(txt_feat)
                        loss += self.lambda_mi * self.mi_criterion(txt_red, hc_feat)
                    if train and self.dc_criterion:
                        txt_red = self._reduce_ensemble(txt_feat)
                        loss += self.lambda_dc * self.dc_criterion(txt_red)

                if train:
                    self.optimizer.zero_grad()
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                # ===== SOFT GATE =====
                p_conf = F.softmax(log_bin, dim=1)[:, 1]
                pred4 = log4.argmax(dim=1)
                pred5 = torch.where(pred4 >= self.CONFUSION_ID, pred4 + 1, pred4)
                pred5[p_conf > self.soft_gate_thr] = self.CONFUSION_ID

                acc = pred5.eq(target).float().mean().item() * 100
                loss_meter.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                all_preds.append(pred5.cpu())
                all_targets.append(target.cpu())

                if i % self.print_freq == 0:
                    ProgressMeter(len(loader), [loss_meter, war_meter], prefix, self.log_txt_path).display(i)

        # ===== METRICS =====
        preds = torch.cat(all_preds)
        targs = torch.cat(all_targets)
        cm = confusion_matrix(targs.numpy(), preds.numpy(), labels=np.arange(5))
        recall = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
        uar = np.mean(recall) * 100

        return war_meter.avg, uar, loss_meter.avg, cm

    def train_epoch(self, loader, epoch):
        return self._run_one_epoch(loader, epoch, train=True)

    def validate(self, loader, epoch):
        return self._run_one_epoch(loader, epoch, train=False)