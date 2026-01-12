# trainer.py
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from utils.utils import AverageMeter, ProgressMeter, get_loss_weight


class Trainer:
    """
    Trainer with:
    - 2-head loss (Confusion vs Non-confusion)
    - Soft-gate inference
    - Automatic threshold sweep on VALIDATION
    """

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
        soft_gate_thr=0.7,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_txt_path = log_txt_path

        self.print_freq = 10

        # -------- 2-head config --------
        self.two_head_loss = two_head_loss
        self.w_bin = w_bin
        self.w_4 = w_4
        self.soft_gate_thr = soft_gate_thr
        self.CONFUSION_ID = 2  # RAER: Confusion

        # -------- MI / DC --------
        self.mi_criterion = mi_criterion
        self.lambda_mi = float(lambda_mi)
        self.dc_criterion = dc_criterion
        self.lambda_dc = float(lambda_dc)

        self.mi_warmup = int(mi_warmup)
        self.mi_ramp = int(mi_ramp)
        self.dc_warmup = int(dc_warmup)
        self.dc_ramp = int(dc_ramp)

        # -------- AMP --------
        self.use_amp = bool(use_amp)
        self.grad_clip = float(grad_clip)
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---------------------------------------------------

    def _maybe_process_ensemble_text_features(self, learnable_text_features):
        if hasattr(self.model, "is_ensemble") and self.model.is_ensemble:
            C = self.model.num_classes
            P = self.model.num_prompts_per_class
            return learnable_text_features.view(C, P, -1).mean(dim=1)
        return learnable_text_features

    # ---------------------------------------------------

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        prefix = f"Train Epoch [{epoch_str}]" if is_train else f"Valid Epoch [{epoch_str}]"
        self.model.train() if is_train else self.model.eval()

        losses = AverageMeter("Loss", ":.4e")
        war_meter = AverageMeter("WAR", ":6.2f") if is_train else None
        progress = ProgressMeter(len(loader), [losses], prefix=prefix, log_txt_path=self.log_txt_path)

        all_targets = []
        all_preds = []
        all_logits_bin = []
        all_logits_4 = []

        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    output, logits_bin, logits_4, learnable_text, hand_text = self.model(
                        images_face, images_body
                    )

                    # -------- Loss --------
                    if self.two_head_loss:
                        target_bin = (target == self.CONFUSION_ID).long()
                        loss_bin = self.criterion(logits_bin, target_bin)

                        mask = target != self.CONFUSION_ID
                        if mask.sum() > 0:
                            target_4 = target[mask]
                            target_4 = torch.where(target_4 > self.CONFUSION_ID, target_4 - 1, target_4)
                            loss_4 = self.criterion(logits_4[mask], target_4)
                        else:
                            loss_4 = torch.tensor(0.0, device=self.device)

                        loss = self.w_bin * loss_bin + self.w_4 * loss_4
                    else:
                        loss = self.criterion(output, target)

                    # -------- MI / DC --------
                    if is_train:
                        proc_text = self._maybe_process_ensemble_text_features(learnable_text)

                        if self.mi_criterion and self.lambda_mi > 0:
                            w = get_loss_weight(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
                            loss += w * self.mi_criterion(proc_text, hand_text)

                        if self.dc_criterion and self.lambda_dc > 0:
                            w = get_loss_weight(int(epoch_str), self.dc_warmup, self.dc_ramp, self.lambda_dc)
                            loss += w * self.dc_criterion(proc_text)

                # -------- Backprop --------
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward() if self.scaler else loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer) if self.scaler else self.optimizer.step()
                    if self.scaler:
                        self.scaler.update()

                losses.update(loss.item(), target.size(0))
                all_targets.append(target.cpu())

                # -------- Predictions --------
                if is_train:
                    if self.two_head_loss:
                        p_conf = F.softmax(logits_bin, dim=1)[:, 1]
                        pred_4 = logits_4.argmax(dim=1)
                        pred = torch.where(pred_4 >= self.CONFUSION_ID, pred_4 + 1, pred_4)
                        pred[p_conf > self.soft_gate_thr] = self.CONFUSION_ID
                    else:
                        pred = output.argmax(dim=1)

                    war_meter.update((pred == target).float().mean().item() * 100, target.size(0))
                    all_preds.append(pred.cpu())
                else:
                    all_logits_bin.append(logits_bin.cpu())
                    all_logits_4.append(logits_4.cpu())

                if i % self.print_freq == 0:
                    progress.display(i)

        # =====================================================
        # METRICS
        # =====================================================
        targets = torch.cat(all_targets).numpy()

        if not is_train and self.two_head_loss:
            logits_bin = torch.cat(all_logits_bin)
            logits_4 = torch.cat(all_logits_4)

            p_conf = F.softmax(logits_bin, dim=1)[:, 1]
            pred_4 = logits_4.argmax(dim=1)
            pred_4 = torch.where(pred_4 >= self.CONFUSION_ID, pred_4 + 1, pred_4)

            best_uar, best_thr, best_cm = -1, None, None
            for thr in np.arange(0.2, 0.81, 0.05):
                pred = pred_4.clone()
                pred[p_conf > thr] = self.CONFUSION_ID
                cm = confusion_matrix(targets, pred.numpy(), labels=np.arange(5))
                acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
                uar = np.nanmean(acc) * 100
                if uar > best_uar:
                    best_uar, best_thr, best_cm = uar, thr, cm

            war = best_cm.diagonal().sum() / best_cm.sum() * 100
            uar = best_uar

            msg = f"{prefix} | WAR {war:.2f} | UAR {uar:.2f} | best_thr={best_thr:.2f}"
        else:
            preds = torch.cat(all_preds).numpy()
            cm = confusion_matrix(targets, preds, labels=np.arange(5))
            war = war_meter.avg
            acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
            uar = np.nanmean(acc) * 100
            msg = f"{prefix} | WAR {war:.2f} | UAR {uar:.2f}"

        print(msg)
        with open(self.log_txt_path, "a") as f:
            f.write(msg + "\n")

        return war, uar, losses.avg, cm

    # ---------------------------------------------------

    def train_epoch(self, loader, epoch):
        return self._run_one_epoch(loader, str(epoch), is_train=True)

    def validate(self, loader, epoch):
        return self._run_one_epoch(loader, str(epoch), is_train=False)