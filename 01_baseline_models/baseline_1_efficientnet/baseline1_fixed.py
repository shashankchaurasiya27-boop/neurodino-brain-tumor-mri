import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Safely pointing to your new Documents directory to avoid macOS Sandbox blocks
    dataset_dir: str = "/Users/shashankchaurasia/Documents/tumor/dataset_archive2/archive"
    train_dir: str = os.path.join(dataset_dir, "Training")
    test_dir: str = os.path.join(dataset_dir, "Testing")
    save_dir: str = "./results_baseline1"
    
    # Model
    model_name: str = "efficientnet_v2_s"
    num_classes: int = 4
    image_size: int = 224 
    
    # Training
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5
    max_grad_norm: float = 1.0
    
    # Dataloader
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Reproducibility
    seed: int = 42


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def verify_dataset(train_dir: str, test_dir: str):
    train_classes = sorted(os.listdir(train_dir))
    test_classes = sorted(os.listdir(test_dir))
    
    # Filter out hidden filesisuue 
    train_classes = [c for c in train_classes if not c.startswith('.')]
    test_classes = [c for c in test_classes if not c.startswith('.')]
    
    print(f"Classes found: {train_classes}")
    
    train_counts = {c: len([f for f in os.listdir(os.path.join(train_dir, c)) if not f.startswith('.')]) for c in train_classes}
    test_counts = {c: len([f for f in os.listdir(os.path.join(test_dir, c)) if not f.startswith('.')]) for c in test_classes}
    
    print("\nTraining Set:")
    for c, count in train_counts.items():
        print(f"  {c}: {count} images")
    print(f"  Total: {sum(train_counts.values())}")
    
    print("\nTesting Set:")
    for c, count in test_counts.items():
        print(f"  {c}: {count} images")
    print(f"  Total: {sum(test_counts.values())}")



def analyze_dataset(directory: str, max_samples: int = 1000):
    widths, heights, aspect_ratios = [], [], []
    classes = [c for c in os.listdir(directory) if not c.startswith('.')]
    
    image_paths = []
    for c in classes:
        cls_dir = os.path.join(directory, c)
        for img_name in os.listdir(cls_dir):
            if not img_name.startswith('.'):
                image_paths.append(os.path.join(cls_dir, img_name))
                
    # Sample if too large
    if len(image_paths) > max_samples:
        image_paths = random.sample(image_paths, max_samples)
        
    for path in tqdm(image_paths, desc="Analyzing Image Sizes"):
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w/h)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    print(f"Analyzed {len(widths)} images.")
    print(f"Width - Mean: {np.mean(widths):.2f}, Min: {np.min(widths)}, Max: {np.max(widths)}")
    print(f"Height - Mean: {np.mean(heights):.2f}, Min: {np.min(heights)}, Max: {np.max(heights)}")
    print(f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(widths, kde=True, bins=30)
    plt.title("Width Distribution")
    
    plt.subplot(1, 3, 2)
    sns.histplot(heights, kde=True, bins=30)
    plt.title("Height Distribution")
    
    plt.subplot(1, 3, 3)
    sns.histplot(aspect_ratios, kde=True, bins=30)
    plt.title("Aspect Ratio Distribution")
    plt.tight_layout()
    plt.show()



train_transforms = transforms.Compose([
    transforms.Resize((Config.image_size, Config.image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((Config.image_size, Config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(Config: Config):
    train_dataset = datasets.ImageFolder(Config.train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(Config.test_dir, transform=test_transforms)
    
    # For reproducible dataloading
    g = torch.Generator()
    g.manual_seed(Config.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        persistent_workers=Config.persistent_workers,
        prefetch_factor=Config.prefetch_factor,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        persistent_workers=Config.persistent_workers,
        prefetch_factor=Config.prefetch_factor,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return train_loader, test_loader, train_dataset.classes


def build_model(num_classes: int, device: torch.device):
    # Load pretrained ImageNet weights
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    
    # Replace classifier layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    
    # Print architecture and parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: EfficientNetV2-S")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    return model, total_params, trainable_params

# print(model) # Uncomment to see full architecture



def train_model(model, train_loader, test_loader, Config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=device.type in ['cuda'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    print("Starting Training...")
    
    for epoch in range(Config.num_epochs):
        epoch_start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=device.type in ['cuda']):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                optimizer.step()
                
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=device.type in ['cuda']):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
                
        # Metrics Calculation
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{Config.num_epochs} - Time: {epoch_time:.2f}s - "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f} - LR: {current_lr:.6f}")
        
        # Early Stopping & Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(Config.save_dir, "best_model.pth"))
            print(f"[*] Validation loss improved. Saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    return history




def evaluate_model(model, test_loader, class_names, device):
    model.load_state_dict(torch.load(os.path.join(Config.save_dir, "best_model.pth"), weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating Best Model...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Metrics computation
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # ROC and PR AUC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    roc_auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
    pr_auc = average_precision_score(y_true_bin, y_probs, average='macro')
    
    print("-" * 50)
    print("FINAL EVALUATION METRICS")
    print("-" * 50)
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision (Macro):  {prec_macro:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}")
    print(f"ROC-AUC (OvR):      {roc_auc:.4f}")
    print(f"PR-AUC (Macro):     {pr_auc:.4f}")
    print("-" * 50)
    
    # Per-class metrics
    class_report_str = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(class_report_str)
    
    # Save metrics to text and CSV
    with open(os.path.join(Config.save_dir, "classification_report.txt"), "w") as f:
        f.write(class_report_str)
        f.write(f"\n\nMetrics:\nAccuracy: {acc}\nROC-AUC: {roc_auc}\nPR-AUC: {pr_auc}")
        
    metrics_df = pd.DataFrame([{
        "Accuracy": acc, "Precision": prec_macro, "Recall": rec_macro,
        "F1": f1_macro, "Balanced_Accuracy": bal_acc, "MCC": mcc,
        "Kappa": kappa, "ROC_AUC": roc_auc, "PR_AUC": pr_auc
    }])
    metrics_df.to_csv(os.path.join(Config.save_dir, "metrics.csv"), index=False)
    
    return y_true, y_pred, y_probs, y_true_bin



def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), bbox_inches='tight')
    
    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), bbox_inches='tight')
    
    # Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['lr'], color='orange')
    plt.title('Learning Rate Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches='tight')
    plt.show()

def plot_roc_pr_curves(y_true_bin, y_probs, class_names, save_dir):
    plt.figure(figsize=(16, 6))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (OvR)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), bbox_inches='tight')
    
    # PR Curve
    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AUC = {pr_auc:.2f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()



def analyze_errors(test_dataset, y_true, y_pred, y_probs, class_names):
    errors = np.where(y_true != y_pred)[0]
    corrects = np.where(y_true == y_pred)[0]
    
    error_probs = np.max(y_probs[errors], axis=1)
    
    highest_conf_errors = errors[np.argsort(error_probs)[-5:]][::-1]
    lowest_conf_preds = np.argsort(np.max(y_probs, axis=1))[:5]
    
    def plot_samples(indices, title):
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            if i >= 5: break
            img_path, true_label_idx = test_dataset.samples[idx]
            true_label = class_names[true_label_idx]
            pred_label = class_names[y_pred[idx]]
            conf = np.max(y_probs[idx])
            
            img = Image.open(img_path).convert('RGB')
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(f"T: {true_label}\nP: {pred_label}\nConf: {conf:.2f}", fontsize=9)
            plt.axis('off')
        plt.suptitle(title, y=1.1)
        plt.show()
        
    print(f"Total Errors: {len(errors)} / {len(y_true)}")
    if len(corrects) > 0:
        plot_samples(corrects[:5], "Sample Correct Predictions")
    if len(errors) > 0:
        plot_samples(errors[:5], "Sample Incorrect Predictions")
        plot_samples(highest_conf_errors, "Highest Confidence Errors")
    plot_samples(lowest_conf_preds, "Lowest Confidence Predictions")



def performance_analysis(model, device, total_params, trainable_params, save_dir):
    # Dummy input
    dummy_input = torch.randn(1, 3, Config.image_size, Config.image_size).to(device)
    
    # Model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    # Inference Time
    model.eval()
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
            
        # Measure
        for _ in range(100):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            
    avg_latency = np.mean(times) * 1000 # ms
    fps = 1.0 / np.mean(times)
    
    summary = (
        f"--- Performance Analysis ---\n"
        f"Total Parameters: {total_params:,}\n"
        f"Trainable Parameters: {trainable_params:,}\n"
        f"Model Size: {size_all_mb:.2f} MB\n"
        f"Average Inference Latency (Batch=1): {avg_latency:.2f} ms\n"
        f"Frames Per Second (FPS): {fps:.2f} images/sec\n"
    )
    
    print(summary)
    with open(os.path.join(save_dir, "model_summary.txt"), "w") as f:
        f.write(summary)




def main():
    Config = Config()
    os.makedirs(Config.save_dir, exist_ok=True)
    print(f"Dataset securely loaded from: {Config.dataset_dir}")
    seed_everything(Config.seed)
    device = get_device()
    verify_dataset(Config.train_dir, Config.test_dir)
    analyze_dataset(Config.train_dir)
    train_loader, test_loader, class_names = create_dataloaders(Config)
    print(f"Class mapping: {class_names}")
    model, total_params, trainable_params = build_model(Config.num_classes, device)
    history = train_model(model, train_loader, test_loader, Config, device)
    y_true, y_pred, y_probs, y_true_bin = evaluate_model(model, test_loader, class_names, device)
    plot_training_history(history, Config.save_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, Config.save_dir)
    plot_roc_pr_curves(y_true_bin, y_probs, class_names, Config.save_dir)
    analyze_errors(test_loader.dataset, y_true, y_pred, y_probs, class_names)
    performance_analysis(model, device, total_params, trainable_params, Config.save_dir)
    print(f"All artifacts saved to: {os.path.abspath(Config.save_dir)}")
    print("Completed Baseline-1 EfficientNetV2 Training and Evaluation.")
if __name__ == "__main__":
    main()
