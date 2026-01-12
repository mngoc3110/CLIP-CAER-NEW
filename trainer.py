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
        w_bin=1.0,
        w_4=1.0,
        soft_gate_thr=0.7
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
        self.w_bin = w_bin
        self.w_4 = w_4
        self.soft_gate_thr = soft_gate_thr
        self.CONFUSION_ID = 2  # As specified in the guide

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

        # Debug prediction images
        self.debug_predictions_path = "debug_predictions"
        os.makedirs(self.debug_predictions_path, exist_ok=True)

    def _save_debug_image(self, tensor, prediction, target, epoch_str, batch_idx, img_idx):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        epoch_debug_path = os.path.join(self.debug_predictions_path, f"epoch_{epoch_str}")
        os.makedirs(epoch_debug_path, exist_ok=True)
        filename = f"batch_{batch_idx}_img_{img_idx}_pred_{prediction}_true_{target}.png"
        filepath = os.path.join(epoch_debug_path, filename)
        torchvision.utils.save_image(tensor, filepath)

    def _maybe_process_ensemble_text_features(self, learnable_text_features):
        processed = learnable_text_features
        if hasattr(self.model, "is_ensemble") and self.model.is_ensemble:
            num_classes = int(self.model.num_classes)
            num_prompts_per_class = int(self.model.num_prompts_per_class)
            processed = learnable_text_features.view(num_classes, num_prompts_per_class, -1).mean(dim=1)
        return processed

    def _apply_logit_adjustment(self, output):
        if self.class_priors is None:
            return output
        priors = self.class_priors.to(output.device)
        return output - self.logit_adj_tau * torch.log(priors + 1e-12)

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        prefix = f"Train Epoch: [{epoch_str}]" if is_train else f"Valid Epoch: [{epoch_str}]"
        if is_train: self.model.train()
        else: self.model.eval()

        losses, mi_losses, dc_losses = AverageMeter("Loss", ":.4e"), AverageMeter("MI Loss", ":.4e"), AverageMeter("DC Loss", ":.4e")
        war_meter = AverageMeter("WAR", ":6.2f")
        meters = [losses, war_meter]
        if self.mi_criterion: meters.insert(1, mi_losses)
        if self.dc_criterion: meters.insert(2, dc_losses)
        progress = ProgressMeter(len(loader), meters, prefix=prefix, log_txt_path=self.log_txt_path)

        all_preds, all_targets = [], []
        # For threshold sweep in validation
        all_p_conf, all_preds_4_raw = [], [] 

        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                images_face = images_face.to(self.device, non_blocking=True)
                images_body = images_body.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp and self.scaler is not None):
                    output, logits_bin, logits_4, learnable_text_features, hand_crafted_text_features = self.model(images_face, images_body)
                    
                    if self.two_head_loss:
                        # --- Stage 2: 2-Head Loss Calculation ---
                        target_bin = (target == self.CONFUSION_ID).long()
                        mask_non_conf = (target != self.CONFUSION_ID)
                        
                        loss_bin = self.criterion(logits_bin, target_bin)
                        
                        # Only compute 4-class loss for non-confusion samples
                        if mask_non_conf.sum() > 0:
                            target_4 = target[mask_non_conf]
                            # Map original labels (0, 1, 3, 4) to new (0, 1, 2, 3)
                            target_4 = torch.where(target_4 > self.CONFUSION_ID, target_4 - 1, target_4)
                            loss_4 = self.criterion(logits_4[mask_non_conf], target_4)
                        else:
                            loss_4 = torch.tensor(0.0, device=self.device) # No non-confusion samples in batch

                        classification_loss = self.w_bin * loss_bin + self.w_4 * loss_4
                    else:
                        # --- Original 5-class loss ---
                        if is_train: # Apply logit adjustment only during training
                            output = self._apply_logit_adjustment(output)
                        classification_loss = self.criterion(output, target)

                    loss = classification_loss
                    # MI/DC only in training
                    if is_train:
                        processed_learnable = self._maybe_process_ensemble_text_features(learnable_text_features)
                        if self.mi_criterion and self.lambda_mi > 0:
                            mi_weight = get_loss_weight(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
                            mi_loss = self.mi_criterion(processed_learnable, hand_crafted_text_features)
                            loss += mi_weight * mi_loss
                            mi_losses.update(mi_loss.item(), target.size(0))
                        if self.dc_criterion and self.lambda_dc > 0:
                            dc_weight = get_loss_weight(int(epoch_str), self.dc_warmup, self.dc_ramp, self.lambda_dc)
                            dc_loss = self.dc_criterion(processed_learnable)
                            loss += dc_weight * dc_loss
                            dc_losses.update(dc_loss.item(), target.size(0))

                # Backprop
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                # --- Metrics ---
                losses.update(loss.item(), target.size(0))
                all_targets.append(target.cpu())
                
                if not is_train and self.two_head_loss:
                    all_p_conf.append(F.softmax(logits_bin, dim=1)[:, 1].cpu())
                    all_preds_4_raw.append(logits_4.argmax(dim=1).cpu())
                else: # For training or original 5-class mode, use current batch predictions
                    if self.two_head_loss: # During training, still use self.soft_gate_thr for WAR
                        p_conf_batch = F.softmax(logits_bin, dim=1)[:, 1]
                        preds_4_batch = logits_4.argmax(dim=1)
                        final_preds_batch = torch.where(preds_4_batch >= self.CONFUSION_ID, preds_4_batch + 1, preds_4_batch)
                        final_preds_batch[p_conf_batch > self.soft_gate_thr] = self.CONFUSION_ID
                        preds_batch = final_preds_batch
                    else:
                        preds_batch = output.argmax(dim=1)
                    war_meter.update(preds_batch.eq(target).sum().item() / target.size(0) * 100.0, target.size(0))
                    all_preds.append(preds_batch.cpu())

                if i % self.print_freq == 0:
                    progress.display(i)

        # Epoch-level metrics
        all_targets_cat = torch.cat(all_targets)
        
        if not is_train and self.two_head_loss:
            # --- Threshold Sweep for Validation ---
            all_p_conf_cat = torch.cat(all_p_conf)
            all_preds_4_raw_cat = torch.cat(all_preds_4_raw)

            threshold_range = torch.arange(0.2, 0.81, 0.05).tolist() # Sweep from 0.2 to 0.8 with 0.05 step
            
            best_uar_sweep = -1.0
            optimal_thr = self.soft_gate_thr # Default to original if no better found
            best_preds_for_cm = None
            
            for current_thr in threshold_range:
                temp_final_preds = torch.where(all_preds_4_raw_cat >= self.CONFUSION_ID, all_preds_4_raw_cat + 1, all_preds_4_raw_cat)
                temp_final_preds[all_p_conf_cat > current_thr] = self.CONFUSION_ID
                
                temp_cm = confusion_matrix(all_targets_cat.numpy(), temp_final_preds.numpy(), labels=np.arange(5))
                temp_class_acc = temp_cm.diagonal() / (temp_cm.sum(axis=1) + 1e-6)
                temp_uar = np.nanmean(temp_class_acc[~np.isnan(temp_class_acc)]) * 100
                
                if temp_uar > best_uar_sweep:
                    best_uar_sweep = temp_uar
                    optimal_thr = current_thr
                    best_preds_for_cm = temp_final_preds
            
            # Use best sweep results for final metrics
            final_preds = best_preds_for_cm
            uar = best_uar_sweep
            
            # Calculate WAR using optimal threshold
            war = final_preds.eq(all_targets_cat).sum().item() / all_targets_cat.size(0) * 100.0
            cm = confusion_matrix(all_targets_cat.numpy(), final_preds.numpy(), labels=np.arange(5))
            
            logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f} | Optimal THR: {optimal_thr:.2f}")
            with open(self.log_txt_path, "a") as f:
                f.write(f"Current WAR: {war:.3f}\n")
                f.write(f"Current UAR: {uar:.3f}\n")
                f.write(f"Optimal THR: {optimal_thr:.2f}\n")
            
        else: # Original 5-class mode OR training with 2-head loss
            all_preds_cat = torch.cat(all_preds)
            cm = confusion_matrix(all_targets_cat.numpy(), all_preds_cat.numpy(), labels=np.arange(5))
            war = war_meter.avg
            class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
            uar = np.nanmean(class_acc[~np.isnan(class_acc)]) * 100

            logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
            with open(self.log_txt_path, "a") as f:
                f.write(f"Current WAR: {war:.3f}\n")
                f.write(f"Current UAR: {uar:.3f}\n")

        return war, uar, losses.avg, cm

    def train_epoch(self, train_loader, epoch_num):
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)

    def validate(self, val_loader, epoch_num_str="Final"):
        return self._run_one_epoch(val_loader, str(epoch_num_str), is_train=False)