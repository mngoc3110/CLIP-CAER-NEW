# trainer.py
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import torchvision

from utils.utils import AverageMeter, ProgressMeter, get_loss_weight

class Trainer:
    """A class that encapsulates the training and validation logic."""
    def __init__(self, model, criterion, optimizer, scheduler, device,log_txt_path, 
                 mi_criterion=None, lambda_mi=0, 
                 dc_criterion=None, lambda_dc=0,
                 class_priors=None, logit_adj_tau=1.0,
                 mi_warmup=0, mi_ramp=0,
                 dc_warmup=0, dc_ramp=0, use_amp=False, grad_clip=1.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = 10
        self.log_txt_path = log_txt_path
        self.mi_criterion = mi_criterion
        self.lambda_mi = lambda_mi
        self.dc_criterion = dc_criterion
        self.lambda_dc = lambda_dc
        self.class_priors = class_priors
        self.logit_adj_tau = logit_adj_tau
        self.mi_warmup = mi_warmup
        self.mi_ramp = mi_ramp
        self.dc_warmup = dc_warmup
        self.dc_ramp = dc_ramp
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Create directory for saving debug prediction images
        self.debug_predictions_path = 'debug_predictions'
        os.makedirs(self.debug_predictions_path, exist_ok=True)

    def _save_debug_image(self, tensor, prediction, target, epoch_str, batch_idx, img_idx):
        """Saves a single image tensor for debugging, with prediction and target in the filename."""
        # Un-normalize the image
        # These are common normalization values for ImageNet, adjust if yours are different
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)

        # Create a directory for the current epoch if it doesn't exist
        epoch_debug_path = os.path.join(self.debug_predictions_path, f"epoch_{epoch_str}")
        os.makedirs(epoch_debug_path, exist_ok=True)
        
        # Construct filename
        filename = f"batch_{batch_idx}_img_{img_idx}_pred_{prediction}_true_{target}.png"
        filepath = os.path.join(epoch_debug_path, filename)
        
        # Save the image
        torchvision.utils.save_image(tensor, filepath)


    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        """Runs one epoch of training or validation."""
        if is_train:
            self.model.train()
            prefix = f"Train Epoch: [{epoch_str}]"
        else:
            self.model.eval()
            prefix = f"Valid Epoch: [{epoch_str}]"

        losses = AverageMeter('Loss', ':.4e')
        mi_losses = AverageMeter('MI Loss', ':.4e')
        dc_losses = AverageMeter('DC Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        
        progress_meters = [losses, war_meter]
        if self.mi_criterion is not None:
            progress_meters.insert(1, mi_losses)
        if self.dc_criterion is not None:
            progress_meters.insert(2, dc_losses)

        progress = ProgressMeter(
            len(loader), 
            progress_meters, 
            prefix=prefix, 
            log_txt_path=self.log_txt_path  
        )

        all_preds = []
        all_targets = []
        saved_images_count = 0


        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                # Debugging: Print batch information
                if is_train:
                    print(f"--> Batch {i}, Size: {target.size(0)}, Labels: {target.tolist()}")

                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Forward pass
                    output, learnable_text_features, hand_crafted_text_features = self.model(images_face, images_body)
                    
                    # For MI and DC losses, if using prompt ensembling, average the learnable_text_features
                    processed_learnable_text_features = learnable_text_features
                    if hasattr(self.model, 'is_ensemble') and self.model.is_ensemble:
                        num_classes = self.model.num_classes
                        num_prompts_per_class = self.model.num_prompts_per_class
                        # Reshape from (C*P, D) to (C, P, D) and then average over P
                        processed_learnable_text_features = learnable_text_features.view(num_classes, num_prompts_per_class, -1).mean(dim=1)

                    # Apply logit adjustment
                    if self.class_priors is not None and is_train:
                        output = output - self.logit_adj_tau * torch.log(self.class_priors + 1e-12)

                    # Calculate loss
                    classification_loss = self.criterion(output, target)
                    loss = classification_loss

                    if is_train and self.mi_criterion is not None:
                        mi_weight = get_loss_weight(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
                        mi_loss = self.mi_criterion(processed_learnable_text_features, hand_crafted_text_features)
                        loss += mi_weight * mi_loss
                        mi_losses.update(mi_loss.item(), target.size(0))

                    if is_train and self.dc_criterion is not None:
                        dc_weight = get_loss_weight(int(epoch_str), self.dc_warmup, self.dc_ramp, self.lambda_dc)
                        dc_loss = self.dc_criterion(processed_learnable_text_features)
                        loss += dc_weight * dc_loss
                        dc_losses.update(dc_loss.item(), target.size(0))

                if is_train:
                    self.optimizer.zero_grad()
                    if self.use_amp:
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

                # Record metrics
                preds = output.argmax(dim=1)
                correct_preds = preds.eq(target).sum().item()
                acc = (correct_preds / target.size(0)) * 100.0

                losses.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

                if not is_train and saved_images_count < 32:
                    for img_idx in range(images_face.size(0)):
                        if saved_images_count < 32:
                            self._save_debug_image(
                                images_face[img_idx].cpu(),
                                preds[img_idx].item(),
                                target[img_idx].item(),
                                epoch_str,
                                i,
                                img_idx
                            )
                            saved_images_count += 1
                        else:
                            break


                if i % self.print_freq == 0:
                    progress.display(i)
        
        # Calculate epoch-level metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        war = war_meter.avg # Weighted Average Recall (WAR) is just the overall accuracy
        
        # Unweighted Average Recall (UAR)
        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) # Add epsilon to avoid division by zero
        uar = np.nanmean(class_acc) * 100

        logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        with open(self.log_txt_path, 'a') as f:
            f.write('Current WAR: {war:.3f}'.format(war=war) + '\n')
            f.write('Current UAR: {uar:.3f}'.format(uar=uar) + '\n')
        return war, uar, losses.avg, cm
        
    def train_epoch(self, train_loader, epoch_num):
        """Executes one full training epoch."""
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)
    
    def validate(self, val_loader, epoch_num_str="Final"):
        """Executes one full validation run."""
        return self._run_one_epoch(val_loader, epoch_num_str, is_train=False)