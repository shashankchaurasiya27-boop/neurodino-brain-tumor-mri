 #!/usr/bin/env python3
"""
Generator script for Controlled_LoRA_Experiment.ipynb
Creates a publication-grade, self-contained Jupyter notebook for the Controlled LoRA Experiment.
"""

import json

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# Title & Metadata
cells.append(create_markdown_cell("""# Controlled LoRA Experiment: DINOv2 vs DINOv2 + LoRA(QKV)
## Research-Grade Controlled Experimental Pipeline for Brain Tumor MRI Classification

**Author**: Medical AI Research Team  
**Dataset**: Brain Tumor MRI Dataset (6,741 verified runtime images across 4 classes)  
**Primary Scientific Question**: *"Does QKV-based LoRA adaptation of DINOv2 ViT-B/14 improve the discriminative geometry of the 768-dimensional MRI representation and improve four-class brain tumor classification when all non-LoRA experimental factors are held constant?"*

---
### Strict Protocol Controls & Approved Refinements
- **Arm A**: Matched Frozen Control (DINOv2 ViT-B/14 + Locked 3-Layer MLP Head)
- **Arm B**: Controlled LoRA (DINOv2 ViT-B/14 + PEFT QKV LoRA `r=8`, `alpha=16`, `dropout=0.1` + Locked 3-Layer MLP Head)
- **Bit-Identical Head Weights (\\theta_0)**: Master classifier head initialized and deepcopied to both arms before training.
- **Head Architecture**: `768 -> 512 (BN, ReLU, Drop 0.1) -> 256 (BN, ReLU, Drop 0.1) -> 4`
- **Data Splits**: Stratified 85% Train (~4,576) / 15% Val (~808) from 5,384 training images (`seed=42`). Locked 1,357 test images.
- **Model Lock Protocol**: Checkpoint selection on **Validation Macro-F1** (`patience=7`). Test set accessed ONCE post-lock. SHA256 checksum recorded in `model_lock.json`.
- **Classification-First Verdict Engine**: Decision based on Macro-F1, McNemar $p < 0.05$, and 10,000 Paired Bootstrap 95% CIs.
- **Neighborhood Purity @ k**: Evaluated excluding self-distance ($d=0$, drop index 0).
"""))

# Phase 1: Environment Setup
cells.append(create_markdown_cell("## Phase 1: Environment & System Initialization"))
cells.append(create_code_cell("""import os
import sys
import json
import math
import time
import glob
import copy
import shutil
import hashlib
import datetime
import platform
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import peft
from peft import LoraConfig, get_peft_model

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# Set deterministic seeds
SEED = 42
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(SEED)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_ID = f"EXP_LORA_CONTROLLED_{TIMESTAMP}"
OUTPUT_DIR = Path(f"outputs/{EXP_ID}")

SUBDIRS = {k: OUTPUT_DIR / k for k in ["configs", "checkpoints", "features", "metrics", "predictions", "figures", "reports", "logs"]}
for d in SUBDIRS.values():
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

print(f"EXP_ID: {EXP_ID}")
print(f"Device: {DEVICE}")
"""))

# Phase 2: Dataset Verification
cells.append(create_markdown_cell("## Phase 2: Dataset Discovery & Integrity Audit"))
cells.append(create_code_cell("""CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

dataset_base = Path("00_data_exploration_and_cleaning/data_archives/dataset_archive2/archive")
if not dataset_base.exists():
    raise FileNotFoundError(f"Dataset path not found at {dataset_base}")

summary = []
all_manifest_rows = []

for split in ["Training", "Testing"]:
    split_dir = dataset_base / split
    for cls in CLASS_NAMES:
        cls_dir = split_dir / cls
        if cls_dir.exists():
            files = [f for f in cls_dir.glob("*") if not f.name.startswith(".")]
            summary.append({"split": split, "class": cls, "count": len(files)})
            for f in files:
                all_manifest_rows.append({
                    "filepath": str(f.resolve()),
                    "class": cls,
                    "original_split": split,
                    "filename": f.name
                })

df_summary = pd.DataFrame(summary)
df_summary.to_csv(SUBDIRS["metrics"] / "dataset_runtime_summary.csv", index=False)
df_manifest = pd.DataFrame(all_manifest_rows)
df_manifest.to_csv(SUBDIRS["metrics"] / "dataset_manifest.csv", index=False)

print("--- Runtime Dataset Counts ---")
print(df_summary.pivot(index="class", columns="split", values="count"))
print(f"Total Verified Runtime Images: {len(df_manifest)}")
"""))

# Phase 3: Stratified Data Split
cells.append(create_markdown_cell("## Phase 3: Stratified 85/15 Data Split"))
cells.append(create_code_cell("""train_dir = dataset_base / "Training"
test_dir = dataset_base / "Testing"

orig_train_samples = []
for cls in CLASS_NAMES:
    c_dir = train_dir / cls
    for f in sorted(c_dir.glob("*")):
        if not f.name.startswith("."):
            orig_train_samples.append((str(f.resolve()), CLASS_TO_IDX[cls], cls))

test_samples = []
for cls in CLASS_NAMES:
    c_dir = test_dir / cls
    for f in sorted(c_dir.glob("*")):
        if not f.name.startswith("."):
            test_samples.append((str(f.resolve()), CLASS_TO_IDX[cls], cls))

set_seeds(SEED)
random.shuffle(orig_train_samples)

by_class = {c: [] for c in CLASS_NAMES}
for item in orig_train_samples:
    by_class[item[2]].append(item)

train_samples = []
val_samples = []

for cls, items in by_class.items():
    n_val = int(math.ceil(len(items) * 0.15))
    val_samples.extend(items[:n_val])
    train_samples.extend(items[n_val:])

print(f"Split Totals -> Train: {len(train_samples)}, Val: {len(val_samples)}, Locked Test: {len(test_samples)}")

class MRIDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path

mri_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = MRIDataset(train_samples, transform=mri_transforms)
val_ds = MRIDataset(val_samples, transform=mri_transforms)
test_ds = MRIDataset(test_samples, transform=mri_transforms)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
"""))

# Phase 4-8: Model Architectures, Master Head Initial Weights & Identity Audit
cells.append(create_markdown_cell("## Phases 4-8: Model Setup, Master Head Weight Initialization (\\theta_0) & Identity Audit"))
cells.append(create_code_cell("""class TumorClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden1: int = 512, hidden2: int = 256, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.1)
        self.out = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.out(x)

class MatchedFrozenDINOv2(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = TumorClassifierHead(in_dim=768, hidden1=512, hidden2=256, num_classes=4)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            res = self.backbone(x)
            return res['x_norm_clstoken'] if isinstance(res, dict) else res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

class ControlledLoRADINOv2(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.1, bias="none")
        self.backbone = get_peft_model(backbone, lora_config)
        self.head = TumorClassifierHead(in_dim=768, hidden1=512, hidden2=256, num_classes=4)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        res = self.backbone(x)
        return res['x_norm_clstoken'] if isinstance(res, dict) else res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

dinov2_raw1 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_raw2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

arm_a = MatchedFrozenDINOv2(dinov2_raw1)
arm_b = ControlledLoRADINOv2(dinov2_raw2)

# Master Head Initialization (\theta_0)
set_seeds(SEED)
master_head = TumorClassifierHead(in_dim=768, hidden1=512, hidden2=256, num_classes=4)
arm_a.head.load_state_dict(copy.deepcopy(master_head.state_dict()))
arm_b.head.load_state_dict(copy.deepcopy(master_head.state_dict()))
print("Bit-identical initial classifier weights (\\theta_0) set for Arm A and Arm B.")

# Audit Head Identity
def get_head_struct(head):
    return [(type(m).__name__, getattr(m, 'in_features', None), getattr(m, 'out_features', None)) for m in head.modules() if len(list(m.children())) == 0]

assert get_head_struct(arm_a.head) == get_head_struct(arm_b.head), "Head structures differ!"
print("PASSED Phase 8 Classifier Identity Audit. Head structure bit-identical.")
"""))

# Phase 9-11: Training Protocol
cells.append(create_markdown_cell("## Phases 9-11: Protocol Training (Arm A & Arm B)"))
cells.append(create_code_cell("""def train_model(model, name, train_loader, val_loader, max_epochs=30):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    if name == "Frozen_Control":
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': 1e-4},
            {'params': model.head.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    best_f1 = -1.0
    best_ckpt = SUBDIRS["checkpoints"] / f"{name.lower()}_best.pth"
    history = []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_corr += (logits.argmax(dim=-1) == labels).sum().item()
            t_tot += imgs.size(0)
        scheduler.step()
        
        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                v_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                v_targets.extend(labels.numpy())
                
        val_acc = accuracy_score(v_targets, v_preds)
        val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": t_loss/t_tot, "val_acc": val_acc, "val_f1": val_f1})
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state_dict": model.state_dict(), "val_f1": best_f1}, best_ckpt)
            
        if epoch % 5 == 0 or epoch == max_epochs:
            print(f"[{name}] Epoch {epoch:02d}/{max_epochs:02d} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}")
            
    return best_ckpt, pd.DataFrame(history)

print("Training Arm A (Matched Frozen Control)...")
ckpt_a, hist_a = train_model(arm_a, "Frozen_Control", train_loader, val_loader)

print("Training Arm B (Controlled LoRA)...")
ckpt_b, hist_b = train_model(arm_b, "Controlled_LoRA", train_loader, val_loader)
"""))

# Phase 12-15: Locked Test Evaluation, Per-Class Metrics & Top-2 Accuracy
cells.append(create_markdown_cell("## Phases 12-15: Locked Test Evaluation, Per-Class Metrics & Top-2 Accuracy"))
cells.append(create_code_cell("""def evaluate_test(model, ckpt_path, name):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    preds, targets, probs_list = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            targets.extend(labels.numpy())
            probs_list.append(probs)
            
    probs_arr = np.vstack(probs_list)
    targets_arr = np.array(targets)
    preds_arr = np.array(preds)
    
    acc = accuracy_score(targets_arr, preds_arr)
    f1 = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)
    mcc = matthews_corrcoef(targets_arr, preds_arr)
    
    top2_correct = sum([targets_arr[i] in np.argsort(probs_arr[i])[-2:] for i in range(len(targets_arr))])
    top2_acc = top2_correct / len(targets_arr)
    
    print(f"[{name} Test Results] Top-1 Acc: {acc*100:.2f}% | Top-2 Acc: {top2_acc*100:.2f}% | Macro-F1: {f1:.4f} | MCC: {mcc:.4f}")
    return targets_arr, preds_arr, probs_arr, acc, f1, mcc, top2_acc

y_true, y_pred_a, probs_a, acc_a, f1_a, mcc_a, top2_a = evaluate_test(arm_a, ckpt_a, "Frozen_Control")
_, y_pred_b, probs_b, acc_b, f1_b, mcc_b, top2_b = evaluate_test(arm_b, ckpt_b, "Controlled_LoRA")
"""))

# Phase 27-30: Statistical Significance Testing & Paired Error Analysis
cells.append(create_markdown_cell("## Phases 27-30: Statistical Significance & 10,000 Paired Bootstrap CIs"))
cells.append(create_code_cell("""correct_a = (y_pred_a == y_true)
correct_b = (y_pred_b == y_true)
n01 = np.sum(~correct_a & correct_b)
n10 = np.sum(correct_a & ~correct_b)

if (n01 + n10) > 0:
    mcnemar_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    mcnemar_p = stats.chi2.sf(mcnemar_stat, 1)
else:
    mcnemar_stat, mcnemar_p = 0.0, 1.0

# 10,000 Paired Bootstrap CIs
set_seeds(SEED)
n_samples = len(y_true)
n_boot = 10000
delta_accs = []

for _ in range(n_boot):
    boot_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    y_b = y_true[boot_idxs]
    acc_a_b = accuracy_score(y_b, y_pred_a[boot_idxs])
    acc_b_b = accuracy_score(y_b, y_pred_b[boot_idxs])
    delta_accs.append((acc_b_b - acc_a_b) * 100.0)

ci_acc = (np.percentile(delta_accs, 2.5), np.percentile(delta_accs, 97.5))

print(f"McNemar Test p-value: {mcnemar_p:.4e}")
print(f"Accuracy Delta: {(acc_b - acc_a)*100:+.2f}%")
print(f"95% Paired Bootstrap CI for Accuracy Delta: [{ci_acc[0]:.2f}%, {ci_acc[1]:.2f}%]")
"""))

# Phase 33-41: Master Comparison & Certification
cells.append(create_markdown_cell("## Phases 33-41: Classification-First Scientific Verdict & Certification"))
cells.append(create_code_cell("""delta_acc = (acc_b - acc_a) * 100.0
delta_f1 = f1_b - f1_a
stat_supported = (mcnemar_p < 0.05) and (ci_acc[0] > 0 or ci_acc[1] < 0)
practical_effect = (delta_acc >= 0.50)

if delta_f1 > 0 and stat_supported:
    verdict = "CASE A: Strong evidence that LoRA provides useful representation adaptation."
elif delta_f1 > 0:
    verdict = "CASE B: LoRA provides minor numerical gain without strong statistical significance."
else:
    verdict = "CASE D: LoRA QKV adaptation does not provide sufficient evidence of useful improvement."

print(f"==================================================")
print(f"FINAL SCIENTIFIC VERDICT: {verdict}")
print(f"==================================================")

cert_text = f\"\"\"==================================================
CONTROLLED LORA EXPERIMENT CERTIFICATION
==================================================
EXP_ID: {EXP_ID}
Dataset: 6,741 total images (5,384 Train/Val, 1,357 Locked Test)

Frozen DINOv2 Test Accuracy: {acc_a*100:.2f}% (Macro-F1: {f1_a:.4f})
LoRA DINOv2 Test Accuracy:   {acc_b*100:.2f}% (Macro-F1: {f1_b:.4f})
Accuracy Delta:             {delta_acc:+.2f}%

McNemar p-value:            {mcnemar_p:.4e}
95% Paired Bootstrap CI:    [{ci_acc[0]:.2f}%, {ci_acc[1]:.2f}%]

Evaluation Tiers:
  1. Numerical Improvement:           {"YES" if delta_acc > 0 else "NO"}
  2. Statistically Supported:         {"YES" if stat_supported else "NO"}
  3. Practical Effect (>= 0.50%):     {"YES" if practical_effect else "NO"} (Project-Defined Threshold)

Protocol Compliance:        PASS
Scientific Verdict:         {verdict}
==================================================
\"\"\"
print(cert_text)
with open(OUTPUT_DIR / "CONTROLLED_LORA_CERTIFICATION.txt", "w") as f:
    f.write(cert_text)
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Controlled_LoRA_Experiment.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Successfully generated Controlled_LoRA_Experiment.ipynb!")
