# utils.py
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import yaml
from types import SimpleNamespace
from sklearn.metrics import confusion_matrix
import tqdm
from collections import Counter

def get_loss_weight(epoch, warmup_epochs, ramp_up_epochs, final_weight):
    """Calculates the weight for a loss based on a warmup and ramp-up schedule."""
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + ramp_up_epochs:
        return final_weight * (epoch - warmup_epochs) / ramp_up_epochs
    else:
        return final_weight

def get_class_counts(annotation_file):
    """Reads an annotation file and returns the number of samples for each class."""
    labels = []
    with open(annotation_file, 'r') as f:
        for line in f:
            labels.append(int(line.strip().split()[2]))
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    
    # Sort by class index and get just the counts
    sorted_counts = [class_counts[i] for i in sorted(class_counts)]
    
    return sorted_counts

def slerp(v1, v2, t, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation between two vectors."""
    v1_norm = v1 / torch.norm(v1, p=2, dim=-1, keepdim=True)
    v2_norm = v2 / torch.norm(v2, p=2, dim=-1, keepdim=True)
    
    dot = (v1_norm * v2_norm).sum(dim=-1)
    
    # if dot is > 1, then clamp it to 1
    dot = torch.clamp(dot, -1.0, 1.0)

    # if vectors are too close, use linear interpolation
    if torch.all(dot > DOT_THRESHOLD):
        return (1 - t) * v1 + t * v2

    # calculate the angle between the vectors
    theta = torch.acos(dot) * t
    v3 = v2 - dot.unsqueeze(-1) * v1
    v3 = v3 / torch.norm(v3, p=2, dim=-1, keepdim=True)
    
    return (torch.cos(theta.unsqueeze(-1)) * v1) + (torch.sin(theta.unsqueeze(-1)) * v3)


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_metrics = np.zeros((self.total_epoch, 4), dtype=np.float32)  # [epoch, train_war/train_uar/val_war/val_uar]

    def update(self, idx, train_loss, train_war, train_uar, val_loss, val_war, val_uar):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_metrics[idx, 0] = train_war
        self.epoch_metrics[idx, 1] = train_uar
        self.epoch_metrics[idx, 2] = val_war
        self.epoch_metrics[idx, 3] = val_uar
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'Training and Validation Metrics'
        dpi = 100
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig, ax1 = plt.subplots(figsize=figsize)
        x_axis = np.array([i for i in range(self.current_epoch)])

        # Plot Losses on the first y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(x_axis, self.epoch_losses[:self.current_epoch, 0], color='tab:red', linestyle='-', label='Train Loss')
        ax1.plot(x_axis, self.epoch_losses[:self.current_epoch, 1], color='tab:orange', linestyle='-', label='Valid Loss')
        ax1.tick_params(axis='y')
        
        # Create a second y-axis for the accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)')
        ax2.plot(x_axis, self.epoch_metrics[:self.current_epoch, 0], color='tab:green', linestyle='--', label='Train WAR')
        ax2.plot(x_axis, self.epoch_metrics[:self.current_epoch, 1], color='tab:blue', linestyle='--', label='Train UAR')
        ax2.plot(x_axis, self.epoch_metrics[:self.current_epoch, 2], color='tab:purple', linestyle='--', label='Valid WAR')
        ax2.plot(x_axis, self.epoch_metrics[:self.current_epoch, 3], color='tab:cyan', linestyle='--', label='Valid UAR')
        ax2.tick_params(axis='y')
        ax2.set_ylim(0, 100)

        # Add a single legend for all lines
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=legend_fontsize)
        
        fig.tight_layout()
        plt.title(title, fontsize=20)
        plt.grid()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

def plot_confusion_matrix(cm, classes, normalize=True, title='confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


def computer_uar_war(val_loader, model, device, class_names, log_confusion_matrix_path, log_txt_path, title="Confusion Matrix"):
    model.eval()
    all_predicted = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images_face, images_body, target) in enumerate(tqdm.tqdm(val_loader, desc="Calculating Metrics")):
            images_face = images_face.to(device)
            images_body = images_body.to(device)
            target = target.to(device)

            output = model(images_face, images_body)
            predicted = output.argmax(dim=1)
            
            all_predicted.append(predicted.cpu())
            all_targets.append(target.cpu())

    all_predicted = torch.cat(all_predicted, 0)
    all_targets = torch.cat(all_targets, 0)


    correct = (all_predicted == all_targets).sum().item()
    war = 100. * correct / len(val_loader.dataset)
    

    _confusion_matrix = confusion_matrix(all_targets.numpy(), all_predicted.numpy())
    np.set_printoptions(precision=4)

    class_recall = _confusion_matrix.diagonal() / _confusion_matrix.sum(axis=1)
    class_recall[np.isnan(class_recall)] = 0 
    uar = np.mean(class_recall) * 100.0

    # 4. 打印和记录结果
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm_percent = normalized_cm * 100
    list_diag_percent = np.diag(normalized_cm_percent)

    print("\n--- Evaluation Results ---")
    print(f"Confusion Matrix Diag (%): {list_diag_percent}")
    print(f"UAR: {uar:.2f}%")
    print(f"WAR (Accuracy): {war:.2f}%")
    print("--------------------------\n")

    # 5. 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(normalized_cm_percent, classes=class_names, normalize=True, title=title) # normalize=False因为我们已经手动归一化
    plt.savefig(log_confusion_matrix_path)
    plt.close()
    
    # 6. 写入日志文件
    with open(log_txt_path, 'a') as f:
        f.write('************************\n')
        f.write("Final Evaluation Results:\n")
        f.write("Confusion Matrix Diag (%):\n")
        f.write(str(list_diag_percent.tolist()) + '\n')
        f.write(f'UAR: {uar:.2f}%\n')
        f.write(f'WAR (Accuracy): {war:.2f}%\n')
        f.write('************************\n')
    return uar, war
