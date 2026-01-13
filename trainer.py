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
    """Encapsulates the training and validation logic."""

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
        w_bin=0.5,
        w_4=0.5,
        sweep_range=None,
        conf_recall_min=0.0
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_txt_path = log_txt_path
        self.print_freq = 10

        # Loss & Imbalance
        self.two_head_loss = two_head_loss
        self.w_aux_bin = w_bin
        self.w_aux_4 = w_4
        self.CONFUSION_ID = 2
        self.sweep_range = sweep_range if sweep_range else [0.2, 0.81, 0.05]
        self.conf_recall_min = conf_recall_min

        # MI / DC
        self.mi_criterion = mi_criterion
        self.lambda_mi = float(lambda_mi) if lambda_mi is not None else 0.0
        self.dc_criterion = dc_criterion
        self.lambda_dc = float(lambda_dc) if lambda_dc is not None else 0.0
        self.mi_warmup = int(mi_warmup)
        self.mi_ramp = int(mi_ramp)
        self.dc_warmup = int(dc_warmup)
        self.dc_ramp = int(dc_ramp)

        # Logit adjustment
        self.logit_adj_tau = float(logit_adj_tau)
        self.class_priors = None
        if class_priors is not None:
            if not torch.is_tensor(class_priors):
                class_priors = torch.tensor(class_priors, dtype=torch.float32)
            class_priors = class_priors.float()
            if class_priors.sum().item() > 1.0 + 1e-6:
                class_priors = class_priors / (class_priors.sum() + 1e-12)
            self.class_priors = class_priors.view(1, -1)

        # AMP
        self.use_amp = bool(use_amp)
        self.grad_clip = float(grad_clip)
        self._amp_device = "cuda" if (torch.cuda.is_available() and str(self.device).startswith("cuda")) else "cpu"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self._amp_device == "cuda" else None

    def _calculate_fusion_outputs(self, logits_bin, logits_4):
        """
        Calculates the 5-class fused logits and probabilities using Mixture-of-Experts style fusion.
        """
        batch_size = logits_bin.shape[0]
        num_classes = 5
        
        # --- Get probabilities from heads ---
        probs_bin = F.softmax(logits_bin, dim=1)
        p_conf = probs_bin[:, 1]  # P(is_confusion)
        p_non_conf = probs_bin[:, 0] # P(is_not_confusion)
        
        probs_4 = F.softmax(logits_4, dim=1)
        
        # --- Combine into a 5-class probability distribution ---
        # 1. Create a placeholder with the same dtype and device as the input logits to be AMP-compatible
        final_probs_5 = torch.zeros(batch_size, num_classes, device=logits_4.device, dtype=logits_4.dtype)
        
        # 2. Scatter the 4-class probabilities into the 5-class tensor
        # Create an index mapping: 0->0, 1->1, 2->3, 3->4
        idx_map_4_to_5 = [i for i in range(num_classes) if i != self.CONFUSION_ID]
        final_probs_5[:, idx_map_4_to_5] = probs_4
        
        # 3. Fuse based on the binary head's prediction
        # Weight the 4-class distribution by P(is_not_confusion)
        fused_probs = p_non_conf.unsqueeze(1) * final_probs_5
        
        # Add the probability for the 'Confusion' class, weighted by P(is_confusion)
        fused_probs[:, self.CONFUSION_ID] += p_conf
        
        # --- Convert fused probabilities back to logits for CrossEntropyLoss ---
        final_logits_5 = torch.log(fused_probs + 1e-12)
        
        return final_logits_5

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        prefix = f"Train Epoch: [{epoch_str}]" if is_train else f"Valid Epoch: [{epoch_str}]"
        self.model.train(is_train)

        losses = AverageMeter("Loss", ":.4e")
        war_meter = AverageMeter("WAR", ":6.2f") if is_train else None # Only for train
        progress = ProgressMeter(len(loader), [losses, war_meter] if war_meter else [losses], prefix=prefix, log_txt_path=self.log_txt_path)

        all_targets_cpu, all_preds_cpu = [], []
        all_logits_bin_cpu, all_logits_4_cpu = [], [] # For sweep

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                images_face, images_body, target = images_face.to(self.device), images_body.to(self.device), target.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp and self.scaler is not None):
                    output_5_class_orig, logits_bin, logits_4, learnable_text_features, _ = self.model(images_face, images_body)
                    
                    if self.two_head_loss:
                        final_logits_5 = self._calculate_fusion_outputs(logits_bin, logits_4)
                        loss = self.criterion(final_logits_5, target)
                        if is_train:
                            target_bin = (target == self.CONFUSION_ID).long()
                            loss_aux_bin = self.criterion(logits_bin, target_bin)
                            mask_non_conf = (target != self.CONFUSION_ID)
                            if mask_non_conf.sum() > 0:
                                target_4 = target[mask_non_conf]
                                target_4 = torch.where(target_4 > self.CONFUSION_ID, target_4 - 1, target_4)
                                loss_aux_4 = self.criterion(logits_4[mask_non_conf], target_4)
                            else:
                                loss_aux_4 = 0.0
                            loss = loss + self.w_aux_bin * loss_aux_bin + self.w_aux_4 * loss_aux_4
                    else:
                        if is_train: output_5_class_orig = self._apply_logit_adjustment(output_5_class_orig)
                        loss = self.criterion(output_5_class_orig, target)
                
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip > 0: self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                losses.update(loss.item(), target.size(0))
                all_targets_cpu.append(target.cpu())

                if not is_train and self.two_head_loss:
                    all_logits_bin_cpu.append(logits_bin.cpu())
                    all_logits_4_cpu.append(logits_4.cpu())
                else:
                    preds = (self._calculate_fusion_outputs(logits_bin, logits_4) if self.two_head_loss else output_5_class_orig).argmax(dim=1)
                    war_meter.update(preds.eq(target).sum().item() / target.size(0) * 100.0, target.size(0))
                    all_preds_cpu.append(preds.cpu())

                if i % self.print_freq == 0: progress.display(i)
        
        all_targets_cat = torch.cat(all_targets_cpu)
        
        if not is_train and self.two_head_loss:
            all_logits_bin_cat, all_logits_4_cat = torch.cat(all_logits_bin_cpu), torch.cat(all_logits_4_cpu)
            p_conf = F.softmax(all_logits_bin_cat, dim=1)[:, 1]
            preds_4 = all_logits_4_cat.argmax(dim=1)
            mapped_preds_4 = torch.where(preds_4 >= self.CONFUSION_ID, preds_4 + 1, preds_4)

            thresholds = np.arange(self.sweep_range[0], self.sweep_range[1], self.sweep_range[2]).tolist()
            best_uar, optimal_thr, best_cm = -1.0, 0.5, None

            for thr in thresholds:
                final_preds_for_thr = mapped_preds_4.clone()
                final_preds_for_thr[p_conf > thr] = self.CONFUSION_ID
                
                cm_thr = confusion_matrix(all_targets_cat.numpy(), final_preds_for_thr.numpy(), labels=np.arange(5))
                class_acc_thr = cm_thr.diagonal() / (cm_thr.sum(axis=1) + 1e-6)
                conf_recall = class_acc_thr[self.CONFUSION_ID] * 100.0
                
                if conf_recall < self.conf_recall_min: continue

                uar_thr = np.nanmean(class_acc_thr[~np.isnan(class_acc_thr)]) * 100.0
                if uar_thr > best_uar:
                    best_uar, optimal_thr, best_cm = uar_thr, thr, cm_thr
            
            if best_cm is None: best_cm = confusion_matrix(all_targets_cat.numpy(), mapped_preds_4.numpy(), labels=np.arange(5)) # Fallback
            
            uar, cm = best_uar, best_cm
            war = 100.0 * cm.diagonal().sum() / cm.sum()
            class_recalls = 100.0 * cm.diagonal() / (cm.sum(axis=1) + 1e-6)

            log_msg = f"{prefix} * WAR: {war:.2f}% | UAR: {uar:.2f}% | Optimal THR: {optimal_thr:.2f}"
            recall_str = " | Recalls: " + " ".join([f"{r:.1f}" for r in class_recalls])
            log_msg += recall_str
        else:
            all_preds_cat = torch.cat(all_preds_cpu)
            cm = confusion_matrix(all_targets_cat.numpy(), all_preds_cat.numpy(), labels=np.arange(5))
            war = war_meter.avg
            class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
            uar = np.nanmean(class_acc[~np.isnan(class_acc)]) * 100.0
            log_msg = f"{prefix} * WAR: {war:.2f}% | UAR: {uar:.2f}%"

        logging.info(log_msg)
        with open(self.log_txt_path, "a") as f:
            f.write(log_msg + "\n")
        
        return war, uar, losses.avg, cm

    def train_epoch(self, train_loader, epoch_num):
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)

    def validate(self, val_loader, epoch_num_str="Final"):
        return self._run_one_epoch(val_loader, str(epoch_num_str), is_train=False)
