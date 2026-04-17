# This script trains and evaluates MRI classification experiments for CSC2210
# Credit  Daniel Rafique for baseline early fusion 2 channel
# File has been extensively expanded for new work for CS2210
import os
import time
import argparse
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

import ResNet_2210 as resnet

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fusion', type=str, default='early',
                        choices=['early', 'late', 'late_opt', 'single'])
    parser.add_argument('--modalities', type=str, default='t1_flair',
                        choices=['t1', 'flair', 't1_flair'])
    parser.add_argument('--dtype', type=str, default='fp32',
                        choices=['fp32', 'amp'])
    parser.add_argument('--input_size', type=str, default='full',
                        choices=['full', 'small'])
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size default: 2')
    parser.add_argument('--use_checkpoint', action='store_true',
                    help='Use activation checkpointing in the memory-optimized late fusion model')
    parser.add_argument('--freeze_until', type=str, default='layer2',
                    choices=['none', 'layer1', 'layer2', 'layer3'],
                    help='How far to freeze encoder layers in late_opt')
    return parser.parse_args()

def validate_args(args):
    if args.fusion == 'single' and args.modalities not in ['t1', 'flair']:
        raise ValueError("fusion='single' requires modalities='t1' or 'flair'")

    if args.fusion in ['early', 'late', 'late_opt'] and args.modalities != 't1_flair':
        raise ValueError("fusion='early', 'late', or 'late_opt' requires modalities='t1_flair'")

def pad_or_crop_tensor(tensor, target_shape):
    """
    Pads or center-crops a 4D tensor to target shape (C, D, H, W)

    Args:
        tensor: torch.Tensor of shape (C, D, H, W)
        target_shape: tuple (C, D, H, W)

    Returns:
        tensor resized to target_shape
    """

    c, d, h, w = tensor.shape
    tc, td, th, tw = target_shape

    # Safety check
    if c != tc:
        raise ValueError(f"Channel mismatch: got {c}, expected {tc}")

    # --------------------------
    # CENTER CROP (if too big)
    # --------------------------
    start_d = max((d - td) // 2, 0)
    start_h = max((h - th) // 2, 0)
    start_w = max((w - tw) // 2, 0)

    tensor = tensor[
        :,
        start_d:start_d + min(td, d),
        start_h:start_h + min(th, h),
        start_w:start_w + min(tw, w)
    ]

    # --------------------------
    # SYMMETRIC PAD (if too small)
    # --------------------------
    _, d, h, w = tensor.shape

    pad_d = td - d
    pad_h = th - h
    pad_w = tw - w

    # split padding on both sides
    pad_d_front = max(pad_d // 2, 0)
    pad_d_back  = max(pad_d - pad_d_front, 0)

    pad_h_top   = max(pad_h // 2, 0)
    pad_h_bottom= max(pad_h - pad_h_top, 0)

    pad_w_left  = max(pad_w // 2, 0)
    pad_w_right = max(pad_w - pad_w_left, 0)

    # PyTorch pad order:
    # (W_left, W_right, H_top, H_bottom, D_front, D_back)
    pad = [
        pad_w_left, pad_w_right,
        pad_h_top, pad_h_bottom,
        pad_d_front, pad_d_back
    ]

    tensor = F.pad(tensor, pad, mode='constant', value=0)
    return tensor

def stratified_split(dataset, test_size=0.2, val_size=0.2, seed=42, subset_frac=None):
    """
    Stratified split of the dataset with optional subsampling.
    """
    # Extract labels
    labels = [int(sample[2].item()) for sample in dataset]

    # Optionally subsample for testing
    if subset_frac is not None:
        dataset, _, labels, _ = train_test_split(
            dataset, labels, train_size=subset_frac, stratify=labels, random_state=seed
        )

    # First split into train+val and test
    trainval_data, test_data, trainval_labels, test_labels = train_test_split(
        dataset, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    # Then split train+val into train and val
    val_rel_size = val_size / (1 - test_size)  # Adjust val size relative to train+val
    train_data, val_data, _, _ = train_test_split(
        trainval_data, trainval_labels, test_size=val_rel_size, stratify=trainval_labels, random_state=seed
    )

    return train_data, val_data, test_data


class CustomMRIDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.modalities = args.modalities
        self.input_size = args.input_size

        # Determine number of channels
        if self.modalities in ['t1', 'flair']:
            self.in_channels = 1
        else:
            self.in_channels = 2

        # Set target shape based on input size
        if self.input_size == 'small':
            spatial_shape = (128, 128, 128)
        else:
            spatial_shape = (160, 256, 256)

        self.target_shape = (self.in_channels, *spatial_shape)

    def __getitem__(self, idx):
        image, tabular, label, *_ = self.data[idx] # ignore patient id
        if image.shape[0] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {image.shape[0]}")
        image = pad_or_crop_tensor(image, self.target_shape)
        return image, tabular, label

    def __len__(self):
        return len(self.data)

def build_model(args):
    if args.fusion == 'single':
        model = resnet.MONAIResNet3DWithTabular(
            in_channels=1,
            tabular_dim=2,
            use_tabular=True
        )

    elif args.fusion == 'early':
        model = resnet.MONAIResNet3DWithTabular(
            in_channels=2,
            tabular_dim=2,
            use_tabular=True
        )

    elif args.fusion == 'late':
        model = resnet.LateFusionResNet3DWithTabular(
            tabular_dim=2,
            use_tabular=True
        )

    elif args.fusion == 'late_opt':
        freeze_until = None if args.freeze_until == "none" else args.freeze_until

        model = resnet.LateFusionResNet3DWithTabular_MemoryOptimized(
            tabular_dim=2,
            use_tabular=True,
            use_checkpoint=args.use_checkpoint,
            freeze_until=freeze_until
        )

    else:
        raise ValueError(f"Unsupported fusion type: {args.fusion}")

    return model

def find_best_threshold_youden(y_true, y_probs):
    """
    Find the best decision threshold using Youden's J statistic:
        J = sensitivity - false_positive_rate = TPR - FPR

    Returns:
        best_threshold, best_j
    """
    if len(set(y_true)) < 2:
        return 0.5, float("nan")  # fallback if only one class is present

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = float(thresholds[best_idx])
    best_j = float(j_scores[best_idx])
    return best_threshold, best_j

def train_model(model, train_loader, val_loader, 
                best_auc_model_path, best_loss_model_path,
                scheduled_saved_epochs, scheduled_model_paths,
                log_path, figure_name,
                epochs, lr, weight_decay,
                device, dtype):
    
    model.to(device)

    use_amp = (dtype == "amp" and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Logging setup
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')
    log_file.write(
        "timestamp   |   epoch   |   "
        "train_loss   |   train_acc   |   "
        "val_loss   |   val_acc   |   val_precision   |   val_recall   |   val_f1   |   val_auc   |   val_best_thresh   |   val_youden_j   |   "
        "train_time(s)   |   val_time(s)   |   total_time(s)   |   "
        "data_time(s)   |   compute_time(s)\n"
    )

    train_losses = []
    val_losses = []

    min_epoch_to_save = 1

    # Best AUC checkpoint tracking
    best_val_auc = float("nan")
    best_auc_epoch = None
    best_auc_threshold = 0.5
    best_auc_youden_j = float("nan")
    auc_saved_epochs = []

    # Best loss checkpoint tracking
    best_val_loss = float("inf")
    best_loss_epoch = None
    best_loss_threshold = 0.5
    best_loss_youden_j = float("nan")
    loss_saved_epochs = []
    
    # Automated save of models at epochs
    scheduled_thresholds = {}
    scheduled_youden_j = {}
    actual_scheduled_saved_epochs = []

    epoch_times = []
    train_epoch_times = []
    val_epoch_times = []   
    all_data_times = []
    all_compute_times = []
    peak_mem_train = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()       
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        data_times = []
        compute_times = []
        end = time.perf_counter()

        for imgs, tabular, labels in train_loader:
            data_time = time.perf_counter() - end
            data_times.append(data_time)

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_start = time.perf_counter()

            imgs = imgs.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(imgs, tabular).reshape(-1)
                labels = labels.reshape(-1)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.perf_counter() - compute_start
            compute_times.append(compute_time)
            
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            end = time.perf_counter()

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)

        train_time = time.perf_counter() - train_start
        train_epoch_times.append(train_time)
        avg_data_time_epoch = sum(data_times) / len(data_times)
        avg_compute_time_epoch = sum(compute_times) / len(compute_times)

        val_start = time.perf_counter()

        # Validation
        model.eval()
        val_loss = 0.0
        val_labels, val_probs = [], []

        with torch.no_grad():
            for imgs, tabular, labels in val_loader:
                imgs = imgs.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device).float()

                with autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(imgs, tabular).reshape(-1)
                    labels = labels.reshape(-1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs).float().cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)

        # Choose threshold on validation set using Youden's J
        best_threshold, val_youden_j = find_best_threshold_youden(val_labels, val_probs)

        val_preds = (torch.tensor(val_probs) >= best_threshold).numpy().astype(float)

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_losses.append(val_loss)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = float('nan')  # Happens if only one class present

        val_time = time.perf_counter() - val_start
        val_epoch_times.append(val_time)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        all_data_times.extend(data_times)
        all_compute_times.extend(compute_times)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_mem_train = max(peak_mem_train, peak_mem)

        timestamp = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d_%H%M%S")

        print(f"Timestamp: {timestamp} | "
              f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | "
              f"F1: {val_f1:.4f} | AUC: {val_auc:.4f} | "
              f"Best Thresh: {best_threshold:.4f} | J: {val_youden_j:.4f}")
        
        print(f"Train Time: {train_time:.2f}s | Val Time: {val_time:.2f}s | Total: {epoch_time:.2f}s")

        log_file.write(
            f"{timestamp}   |   {epoch+1}   |   "
            f"{train_loss:.4f}   |   {train_acc:.4f}   |   "
            f"{val_loss:.4f}   |   {val_acc:.4f}   |   {val_precision:.4f}   |   {val_recall:.4f}   |   {val_f1:.4f}   |   {val_auc:.4f}   |   {best_threshold:.4f}   |   {val_youden_j:.4f}   |   "
            f"{train_time:.2f}s   |   {val_time:.2f}s   |   {epoch_time:.2f}s   |   "
            f"{avg_data_time_epoch:.4f}s   |   {avg_compute_time_epoch:.4f}s\n"
        )
        log_file.flush()

        current_epoch = epoch + 1

        # Save scheduled checkpoints at epochs

        if current_epoch in scheduled_saved_epochs:
            save_path = scheduled_model_paths[current_epoch]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

            scheduled_thresholds[current_epoch] = best_threshold
            scheduled_youden_j[current_epoch] = val_youden_j
            actual_scheduled_saved_epochs.append(current_epoch)

            print(f"Scheduled checkpoint saved at epoch {current_epoch}")
            print(f"   Path: {save_path}")
            print(f"   Threshold: {best_threshold:.4f} | Youden J: {val_youden_j:.4f}")

        if current_epoch == min_epoch_to_save:
            # Initialize BOTH best auc and val-loss checkpoints at epoch 1
            best_val_loss = val_loss
            best_loss_epoch = current_epoch
            best_loss_threshold = best_threshold
            best_loss_youden_j = val_youden_j

            os.makedirs(os.path.dirname(best_loss_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_loss_model_path)
            loss_saved_epochs.append(current_epoch)

            print(f"Initial BEST LOSS model saved at epoch {current_epoch}")
            print(f"   Val Loss: {best_val_loss:.4f} | Threshold: {best_loss_threshold:.4f}")

            best_val_auc = val_auc
            best_auc_epoch = current_epoch
            best_auc_threshold = best_threshold
            best_auc_youden_j = val_youden_j

            os.makedirs(os.path.dirname(best_auc_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_auc_model_path)
            auc_saved_epochs.append(current_epoch)

            print(f"Initial BEST AUC model saved at epoch {current_epoch}")
            print(f"   Val AUC: {best_val_auc:.4f} | Threshold: {best_auc_threshold:.4f}")

        elif current_epoch > min_epoch_to_save:
            # Update best loss checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_loss_epoch = current_epoch
                best_loss_threshold = best_threshold
                best_loss_youden_j = val_youden_j

                os.makedirs(os.path.dirname(best_loss_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_loss_model_path)
                loss_saved_epochs.append(current_epoch)

                print(f"Best LOSS model updated at epoch {current_epoch}")
                print(f"   Val Loss: {best_val_loss:.4f} | Threshold: {best_loss_threshold:.4f}")

            # Update best AUC checkpoint
            if not math.isnan(val_auc) and (math.isnan(best_val_auc) or val_auc > best_val_auc):
                best_val_auc = val_auc
                best_auc_epoch = current_epoch
                best_auc_threshold = best_threshold
                best_auc_youden_j = val_youden_j

                os.makedirs(os.path.dirname(best_auc_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_auc_model_path)
                auc_saved_epochs.append(current_epoch)

                print(f"Best AUC model updated at epoch {current_epoch}")
                print(f"   Val AUC: {best_val_auc:.4f} | Threshold: {best_auc_threshold:.4f}")

        scheduler.step() # learning rate wieght decay

    plot_and_save_loss_curves(train_losses, val_losses, output_dir="/workspace/project/figures/", filename=figure_name)
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_train_time = sum(train_epoch_times) / len(train_epoch_times)
    avg_val_time = sum(val_epoch_times) / len(val_epoch_times)
    avg_data_time = sum(all_data_times) / len(all_data_times)
    avg_compute_time = sum(all_compute_times) / len(all_compute_times)

    throughput = len(train_loader.dataset) / avg_epoch_time

    print("\n===== TRAIN EFFICIENCY =====")
    print(f"Avg train time: {avg_train_time:.2f} s")
    print(f"Avg val time: {avg_val_time:.2f} s")
    print(f"Avg total epoch time: {avg_epoch_time:.2f} s")
    print(f"Avg data loading time: {avg_data_time:.4f} s")
    print(f"Avg compute time: {avg_compute_time:.4f} s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Peak GPU memory (train): {peak_mem_train:.2f} GB")

    print("\n===== CHECKPOINT SUMMARY =====")
    print(f"Initial save epoch: {min_epoch_to_save}")
    print(f"Best loss epoch: {best_loss_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best loss threshold: {best_loss_threshold:.4f}")
    print(f"Best loss Youden J: {best_loss_youden_j:.4f}")
    print(f"Loss saved epochs: {loss_saved_epochs}")
    print(f"Best AUC epoch: {best_auc_epoch}")
    print(f"Best val AUC: {best_val_auc:.4f}")
    print(f"Best AUC threshold: {best_auc_threshold:.4f}")
    print(f"Best AUC Youden J: {best_auc_youden_j:.4f}")
    print(f"AUC saved epochs: {auc_saved_epochs}")
    print(f"Requested scheduled epochs: {scheduled_saved_epochs}")
    print(f"Actual scheduled saved epochs: {actual_scheduled_saved_epochs}")
    print(f"Scheduled thresholds: {scheduled_thresholds}")
    print(f"Scheduled Youden J: {scheduled_youden_j}")

    log_file.write("\n===== TRAIN EFFICIENCY =====\n")
    log_file.write(f"Avg train time: {avg_train_time}\n")
    log_file.write(f"Avg val time: {avg_val_time}\n")
    log_file.write(f"Avg total epoch time: {avg_epoch_time}\n")                     
    log_file.write(f"Avg data loading time: {avg_data_time}\n")
    log_file.write(f"Avg compute time: {avg_compute_time}\n")
    log_file.write(f"Throughput: {throughput}\n")
    log_file.write(f"Peak GPU memory (train): {peak_mem_train}\n")

    log_file.write("\n===== CHECKPOINT SUMMARY =====\n")
    log_file.write(f"Initial save epoch: {min_epoch_to_save}\n")
    log_file.write(f"Best loss epoch: {best_loss_epoch}\n")
    log_file.write(f"Best val loss: {best_val_loss}\n")
    log_file.write(f"Best loss threshold: {best_loss_threshold}\n")
    log_file.write(f"Best loss Youden J: {best_loss_youden_j}\n")
    log_file.write(f"Loss saved epochs: {loss_saved_epochs}\n")
    log_file.write(f"Best AUC epoch: {best_auc_epoch}\n")
    log_file.write(f"Best val AUC: {best_val_auc}\n")
    log_file.write(f"Best AUC threshold: {best_auc_threshold}\n")
    log_file.write(f"Best AUC Youden J: {best_auc_youden_j}\n")
    log_file.write(f"AUC saved epochs: {auc_saved_epochs}\n")
    log_file.write(f"Requested scheduled epochs: {scheduled_saved_epochs}\n")
    log_file.write(f"Actual scheduled saved epochs: {actual_scheduled_saved_epochs}\n")
    log_file.write(f"Scheduled thresholds: {scheduled_thresholds}\n")
    log_file.write(f"Scheduled Youden J: {scheduled_youden_j}\n")

    return {
        "best_loss_epoch": best_loss_epoch,
        "best_val_loss": best_val_loss,
        "best_loss_threshold": best_loss_threshold,
        "best_loss_youden_j": best_loss_youden_j,
        "loss_saved_epochs": loss_saved_epochs,
        "best_auc_epoch": best_auc_epoch,
        "best_val_auc": best_val_auc,
        "best_auc_threshold": best_auc_threshold,
        "best_auc_youden_j": best_auc_youden_j,
        "auc_saved_epochs": auc_saved_epochs,
        "requested_scheduled_saved_epochs": scheduled_saved_epochs,
        "actual_scheduled_saved_epochs": actual_scheduled_saved_epochs,
        "scheduled_thresholds": scheduled_thresholds,
        "scheduled_youden_j": scheduled_youden_j,
        "avg_train_time_s": avg_train_time,
        "avg_val_time_s": avg_val_time,
        "avg_epoch_time_s": avg_epoch_time,
        "avg_data_time_s": avg_data_time,
        "avg_compute_time_s": avg_compute_time,
        "train_throughput_samples_per_s": throughput,
        "peak_train_gpu_mem_gb": peak_mem_train,
    }

def plot_and_save_loss_curves(train_losses, val_losses, output_dir, filename):
    """
    Plots and saves training and validation loss curves.

    Args:
        train_losses (list or np.array): List of training loss values per epoch.
        val_losses (list or np.array): List of validation loss values per epoch.
        output_dir (str): Path to directory where the plot should be saved.
        filename (str): Name of the file to save the plot (default: "loss_curve.png").
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")

def get_patient_ids(dataset):
    return [sample[3] for sample in dataset]

# Function to evaluate model and save an roc curve on test data 
def evaluate_model(model, dataloader, device, dtype,
                   save_dir, figure_name, log_path, threshold=0.5):
    model.eval()
    use_amp = (dtype == "amp" and device.type == "cuda") 

    latencies = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    y_true = []
    y_pred_probs = []

    with torch.no_grad():
        # warmup 
        for i, (image, tabular, label) in enumerate(dataloader):
            image = image.to(device)
            tabular = tabular.to(device)

            with autocast(device_type="cuda", enabled=use_amp):
                _ = model(image, tabular)

            if i >= 3:
                break

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed inference
        for image, tabular, label in dataloader:
            image = image.to(device)
            tabular = tabular.to(device)
            label = label.to(device)
        
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(image, tabular)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            batch_size = image.shape[0]
            latencies.append(elapsed / batch_size)

            probs = torch.sigmoid(outputs).float().reshape(-1)
            labels_flat = label.reshape(-1)

            y_pred_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels_flat.cpu().numpy().tolist())

    y_pred = [1 if p >= threshold else 0 for p in y_pred_probs]

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = float("nan")

    print("\n===== INFERENCE RESULTS =====")
    print(f"Threshold used: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    avg_latency = sum(latencies) / len(latencies)
    throughput = 1 / avg_latency
    peak_mem_infer = 0.0
    if device.type == "cuda":
        peak_mem_infer = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print("===== INFERENCE EFFICIENCY =====")
    print(f"Avg latency per sample: {avg_latency:.4f} s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Peak GPU memory (inference): {peak_mem_infer:.2f} GB")

    # Plot ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, figure_name)
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
        plt.close()
    except ValueError:
        print("ROC curve could not be generated because only one class was present.")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write(
            f"\n===== INFERENCE RESULTS =====\n"
            f"Threshold used: {threshold}\n"
            f"Accuracy: {accuracy}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1}\n"
            f"AUC: {auc}\n"
            f"TP: {tp}\n"
            f"TN: {tn}\n"
            f"FP: {fp}\n"
            f"FN: {fn}\n"
            f"\n===== INFERENCE EFFICIENCY =====\n"
            f"Avg latency per sample: {avg_latency}\n"
            f"Throughput: {throughput}\n"
            f"Peak GPU memory (inference): {peak_mem_infer}\n"
        )

    return {
        "test_threshold_used": threshold,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auc": auc,
        "test_tp": int(tp),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "infer_latency_s": avg_latency,
        "infer_throughput_samples_per_s": throughput,
        "peak_infer_gpu_mem_gb": peak_mem_infer,
    }
# -------------- Driver Code for Training -------------- ###

if __name__ == "__main__":
    args = get_args()
    validate_args(args)
    scheduled_saved_epochs = [45,60]

    extra = ""
    if args.fusion == "late_opt":
        extra = f"_ckpt{int(args.use_checkpoint)}_freeze{args.freeze_until}"
    exp_name = f"{args.fusion}_{args.modalities}_{args.dtype}_{args.input_size}_bs{args.batch_size}_{args.epochs}{extra}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nNew experiment using device:", device)
    print("Experiment config:", vars(args))

    dataset = resnet.load_dataset(args.modalities, '/workspace/Processed', '/workspace/MRI_data/participants.tsv')
    
    # Split data
    train_set, val_set, test_set = stratified_split(dataset, test_size=0.2, val_size=0.2)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    #print("\nTrain patient IDs:")
    #print(get_patient_ids(train_set))
    #print("\nVal patient IDs:")
    #print(get_patient_ids(val_set))
    #print("\nTest patient IDs:")
    #print(get_patient_ids(test_set))
    
    # Dataloaders
    train_loader = DataLoader(
        CustomMRIDataset(train_set, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    val_loader = DataLoader(
        CustomMRIDataset(val_set, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    test_loader = DataLoader(
        CustomMRIDataset(test_set, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    model = build_model (args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    timestamp = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d_%H%M%S")

    best_auc_model_path = f"/workspace/models/{exp_name}_{timestamp}_best_auc.pkl"
    best_loss_model_path = f"/workspace/models/{exp_name}_{timestamp}_best_loss.pkl"

    scheduled_model_paths = {
        ep: f"/workspace/models/{exp_name}_{timestamp}_epoch{ep}.pkl"
        for ep in scheduled_saved_epochs
    } 

    train_log_path = f"/workspace/project/logs/log_train_{exp_name}_{timestamp}.txt"
    loss_figure_name = f"loss_{exp_name}_{timestamp}.png"

    eval_auc_log_path = f"/workspace/project/logs/log_eval_auc_{exp_name}_{timestamp}.txt"
    eval_auc_figure_name = f"eval_auc_{exp_name}_{timestamp}.png"

    eval_loss_log_path = f"/workspace/project/logs/log_eval_loss_{exp_name}_{timestamp}.txt"
    eval_loss_figure_name = f"eval_loss_{exp_name}_{timestamp}.png"

    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        best_auc_model_path=best_auc_model_path,
        best_loss_model_path=best_loss_model_path,
        scheduled_saved_epochs=scheduled_saved_epochs,
        scheduled_model_paths=scheduled_model_paths,
        log_path=train_log_path,
        figure_name=loss_figure_name,
        epochs=args.epochs,
        lr=1e-4,
        weight_decay=1e-5,
        device=device,
        dtype=args.dtype
    )

    # Evaluate best AUC checkpoint
    model.load_state_dict(
        torch.load(best_auc_model_path, map_location=device, weights_only=True)
    )
    model.eval()

    eval_auc_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        dtype=args.dtype,
        save_dir="/workspace/project/figures",
        figure_name=eval_auc_figure_name,
        log_path=eval_auc_log_path,
        threshold=train_results["best_auc_threshold"],
    )

    # Evaluate best loss checkpoint
    model.load_state_dict(
        torch.load(best_loss_model_path, map_location=device, weights_only=True)
    )
    model.eval()

    eval_loss_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        dtype=args.dtype,
        save_dir="/workspace/project/figures",
        figure_name=eval_loss_figure_name,
        log_path=eval_loss_log_path,
        threshold=train_results["best_loss_threshold"],
    )

    scheduled_eval_results = {}

    for ep in train_results["actual_scheduled_saved_epochs"]:
        ckpt_path = scheduled_model_paths[ep]
        threshold = train_results["scheduled_thresholds"][ep]

        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.eval()

        eval_log_path = f"/workspace/project/logs/log_eval_epoch{ep}_{exp_name}_{timestamp}.txt"
        eval_figure_name = f"eval_epoch{ep}_{exp_name}_{timestamp}.png"

        scheduled_eval_results[f"epoch_{ep}"] = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            dtype=args.dtype,
            save_dir="/workspace/project/figures",
            figure_name=eval_figure_name,
            log_path=eval_log_path,
            threshold=threshold,
        )

    summary = {
        "exp_name": exp_name,
        "fusion": args.fusion,
        "modalities": args.modalities,
        "dtype": args.dtype,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
        **train_results,
        "best_auc_model_test": eval_auc_results,
        "best_loss_model_test": eval_loss_results,
        "scheduled_model_test": scheduled_eval_results,
    }

    os.makedirs("/workspace/project/results", exist_ok=True)
    summary_path = f"/workspace/project/results/{exp_name}_{timestamp}.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved JSON summary to {summary_path}")
    print("Experiment ended.\n")