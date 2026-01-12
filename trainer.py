# trainer.py
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import confusion_matrix

from utils.utils import AverageMeter, ProgressMeter, get_loss_weight


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        log_txt_path,
        mi_criterion=None,
        lambda_mi=0.0,
        dc_criterion=None,
        lambda_dc=0.0,
        class_priors=None,
        logit_adj_tau=1.0,
        mi_warmup=0,
        mi_ramp=0,
        dc_warmup=0,
        dc_ramp=0,
        use_amp=False,
        grad_clip=1.0,
        two_head_loss=False,
        w_bin=1.0,
        w_4=1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_txt_path = log_txt_path

        self.print_freq = 10

        # ---- Two-head ----
        self.two_head_loss = two_head_loss
        self.w_bin = w_bin
        self.w_4 = w_4
        self.CONFUSION_ID = 2  # RAER: Neutral, Enjoyment, Confusion, Fatigue, Distraction

        # ---- MI / DC ----
        self.mi_criterion = mi_criterion
        self.lambda_mi = float(lambda_mi)
        self.dc_criterion = dc_criterion
        self.lambda_dc = float(lambda_dc)

        self.mi_warmup = int(mi_warmup)
        self.mi_ramp = int(mi_ramp)
        self.dc_warmup = int(dc_warmup)
        self.dc_ramp = int(dc_ramp)

        # ---- Logit adjustment (optional) ----
        self.logit_adj_tau = float(logit_adj_tau)
        self.class_priors = None
        if class_priors is not None:
            if not torch.is_tensor(class_priors):
                class_priors = torch.tensor(class_priors, dtype=torch.float32)
            class_priors = class_priors / (class_priors.sum() + 1e-12)
            self.class_priors = class_priors.view(1, -1)

        # ---- AMP ----
        self.use_amp = bool(use_amp)
        self.grad_clip = float(grad_clip)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    # ======================================================
    # Core epoch
    # ======================================================
    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        prefix = f"Train Epoch: [{epoch_str}]" if is_train else f"Valid Epoch: [{epoch_str}]"
        self.model.train() if is_train else self.model.eval()

        losses = AverageMeter("Loss", ":.4e")
        war_meter = AverageMeter("WAR", ":6.2f")

        meters = [losses, war_meter]
        progress = ProgressMeter(len(loader), meters, prefix=prefix, log_txt_path=self.log_txt_path)

        all_targets, all_preds = [], []

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                images_face = images_face.to(self.device, non_blocking=True)
                images_body = images_body.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output, logits_bin, logits_4, learnable_text_features, hand_crafted_text_features = \
                        self.model(images_face, images_body)

                    # ======================================================
                    # LOSS
                    # ======================================================
                    if self.two_head_loss:
                        # ---- binary loss ----
                        target_bin = (target == self.CONFUSION_ID).long()
                        loss_bin = F.cross_entropy(logits_bin, target_bin)

                        # ---- 4-class loss (non-confusion only) ----
                        mask = target != self.CONFUSION_ID
                        if mask.sum() > 0:
                            target_4 = target[mask]
                            target_4 = torch.where(target_4 > self.CONFUSION_ID, target_4 - 1, target_4)
                            loss_4 = F.cross_entropy(logits_4[mask], target_4)
                        else:
                            loss_4 = torch.tensor(0.0, device=self.device)

                        cls_loss = self.w_bin * loss_bin + self.w_4 * loss_4
                    else:
                        cls_loss = self.criterion(output, target)

                    loss = cls_loss

                    # ---- MI / DC ----
                    if is_train:
                        if self.mi_criterion and self.lambda_mi > 0:
                            mi_w = get_loss_weight(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
                            loss += mi_w * self.mi_criterion(learnable_text_features, hand_crafted_text_features)

                        if self.dc_criterion and self.lambda_dc > 0:
                            dc_w = get_loss_weight(int(epoch_str), self.dc_warmup, self.dc_ramp, self.lambda_dc)
                            loss += dc_w * self.dc_criterion(learnable_text_features)

                # ======================================================
                # BACKWARD
                # ======================================================
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
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

                # ======================================================
                # FUSION A – PREDICTION (NO THRESHOLD)
                # ======================================================
                if self.two_head_loss:
                    p_conf = F.softmax(logits_bin, dim=1)[:, 1]          # (N,)
                    p_non = 1.0 - p_conf

                    prob_4 = F.softmax(logits_4, dim=1)                  # (N,4)

                    # expand to 5 classes
                    prob_5_from_4 = torch.zeros((prob_4.size(0), 5), device=prob_4.device)
                    prob_5_from_4[:, 0:2] = prob_4[:, 0:2]
                    prob_5_from_4[:, 3:5] = prob_4[:, 2:4]

                    prob_conf = torch.zeros_like(prob_5_from_4)
                    prob_conf[:, self.CONFUSION_ID] = 1.0

                    prob_5 = p_non.unsqueeze(1) * prob_5_from_4 + p_conf.unsqueeze(1) * prob_conf
                    preds = prob_5.argmax(dim=1)
                else:
                    preds = output.argmax(dim=1)

                # ======================================================
                # METRICS
                # ======================================================
                losses.update(loss.item(), target.size(0))
                war_meter.update((preds == target).float().mean().item() * 100.0, target.size(0))

                all_targets.append(target.cpu())
                all_preds.append(preds.cpu())

                if i % self.print_freq == 0:
                    progress.display(i)

        # ======================================================
        # FINAL METRICS
        # ======================================================
        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)

        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=np.arange(5))
        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
        uar = np.nanmean(class_acc) * 100.0
        war = war_meter.avg

        log_msg = f"{prefix} * WAR: {war:.2f}% | UAR: {uar:.2f}%"
        logging.info(log_msg)
        with open(self.log_txt_path, "a") as f:
            f.write(log_msg + "\n")

        return war, uar, losses.avg, cm

    def train_epoch(self, train_loader, epoch_num):
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)

    def validate(self, val_loader, epoch_num_str="Final"):
        return self._run_one_epoch(val_loader, str(epoch_num_str), is_train=False)