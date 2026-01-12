# trainer.py
import logging
import os

import numpy as np
import torch
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
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_txt_path = log_txt_path

        self.print_freq = 10

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

            # If counts were provided, normalize to probabilities
            if class_priors.sum().item() > 1.0 + 1e-6:
                class_priors = class_priors / (class_priors.sum() + 1e-12)

            # store shape (1, C) for broadcasting against (N, C)
            self.class_priors = class_priors.view(1, -1)

        # AMP
        self.use_amp = bool(use_amp)
        self.grad_clip = float(grad_clip)
        self._amp_device = "cuda" if (torch.cuda.is_available() and str(self.device).startswith("cuda")) else "cpu"

        self.scaler = None
        if self.use_amp and self._amp_device == "cuda":
            # PyTorch 2.6 recommended API
            self.scaler = torch.amp.GradScaler("cuda")

        # Debug prediction images
        self.debug_predictions_path = "debug_predictions"
        os.makedirs(self.debug_predictions_path, exist_ok=True)

    def _save_debug_image(self, tensor, prediction, target, epoch_str, batch_idx, img_idx):
        """Save a single image tensor for debugging."""
        # Un-normalize image (ImageNet stats) - adjust if your pipeline differs
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
        """
        If model uses prompt ensemble with shape (C*P, D), reduce to (C, D)
        for MI/DC by averaging over prompts.
        """
        processed = learnable_text_features
        if hasattr(self.model, "is_ensemble") and self.model.is_ensemble:
            num_classes = int(self.model.num_classes)
            num_prompts_per_class = int(self.model.num_prompts_per_class)
            processed = learnable_text_features.view(num_classes, num_prompts_per_class, -1).mean(dim=1)
        return processed

    def _apply_logit_adjustment(self, output):
        """
        Apply logit adjustment using TRAIN priors.
        IMPORTANT: priors must be moved to output.device.
        """
        if self.class_priors is None:
            return output
        priors = self.class_priors.to(output.device)  # <-- FIX device mismatch
        return output - self.logit_adj_tau * torch.log(priors + 1e-12)

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        if is_train:
            self.model.train()
            prefix = f"Train Epoch: [{epoch_str}]"
        else:
            self.model.eval()
            prefix = f"Valid Epoch: [{epoch_str}]"

        losses = AverageMeter("Loss", ":.4e")
        mi_losses = AverageMeter("MI Loss", ":.4e")
        dc_losses = AverageMeter("DC Loss", ":.4e")
        war_meter = AverageMeter("WAR", ":6.2f")

        meters = [losses, war_meter]
        if self.mi_criterion is not None:
            meters.insert(1, mi_losses)
        if self.dc_criterion is not None:
            meters.insert(2, dc_losses)

        progress = ProgressMeter(len(loader), meters, prefix=prefix, log_txt_path=self.log_txt_path)

        all_preds, all_targets = [], []
        saved_images_count = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                if is_train:
                    print(f"--> Batch {i}, Size: {target.size(0)}, Labels: {target.tolist()}")

                images_face = images_face.to(self.device, non_blocking=True)
                images_body = images_body.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # AMP autocast only meaningful on CUDA
                if self.use_amp and self._amp_device == "cuda":
                    autocast_ctx = torch.amp.autocast("cuda")
                else:
                    autocast_ctx = torch.amp.autocast("cpu", enabled=False)

                with autocast_ctx:
                    output, learnable_text_features, hand_crafted_text_features = self.model(images_face, images_body)

                    # Prompt ensemble reduction for MI/DC
                    processed_learnable_text_features = self._maybe_process_ensemble_text_features(learnable_text_features)

                    # Logit adjustment (apply both train/valid to keep metrics consistent)
                    output = self._apply_logit_adjustment(output)

                    # Base classification loss
                    classification_loss = self.criterion(output, target)
                    loss = classification_loss

                    # MI/DC only in training
                    if is_train and self.mi_criterion is not None and self.lambda_mi > 0:
                        mi_weight = get_loss_weight(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
                        mi_loss = self.mi_criterion(processed_learnable_text_features, hand_crafted_text_features)
                        loss = loss + mi_weight * mi_loss
                        mi_losses.update(mi_loss.item(), target.size(0))

                    if is_train and self.dc_criterion is not None and self.lambda_dc > 0:
                        dc_weight = get_loss_weight(int(epoch_str), self.dc_warmup, self.dc_ramp, self.lambda_dc)
                        dc_loss = self.dc_criterion(processed_learnable_text_features)
                        loss = loss + dc_weight * dc_loss
                        dc_losses.update(dc_loss.item(), target.size(0))

                # Backprop
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.use_amp and self._amp_device == "cuda" and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip and self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip and self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                # Metrics
                preds = output.argmax(dim=1)
                correct = preds.eq(target).sum().item()
                acc = (correct / target.size(0)) * 100.0

                losses.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                all_preds.append(preds.detach().cpu())
                all_targets.append(target.detach().cpu())

                # Save a few debug images during validation
                if (not is_train) and saved_images_count < 32:
                    bs = images_face.size(0)
                    for img_idx in range(bs):
                        if saved_images_count >= 32:
                            break
                        self._save_debug_image(
                            images_face[img_idx].detach().cpu(),
                            int(preds[img_idx].item()),
                            int(target[img_idx].item()),
                            epoch_str,
                            i,
                            img_idx,
                        )
                        saved_images_count += 1

                if i % self.print_freq == 0:
                    progress.display(i)

        # Epoch-level metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        war = war_meter.avg

        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
        uar = float(np.nanmean(class_acc) * 100.0)

        logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        with open(self.log_txt_path, "a") as f:
            f.write(f"Current WAR: {war:.3f}\n")
            f.write(f"Current UAR: {uar:.3f}\n")

        return war, uar, losses.avg, cm

    def train_epoch(self, train_loader, epoch_num):
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)

    def validate(self, val_loader, epoch_num_str="Final"):
        return self._run_one_epoch(val_loader, str(epoch_num_str), is_train=False)