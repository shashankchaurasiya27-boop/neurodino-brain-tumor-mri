#!/usr/bin/env python3
"""
Notebook E Generator: Notebook_E_Contrastive_Adapter.ipynb
================================================================================
Generates an interactive Jupyter Notebook implementing the 
Supervised Contrastive Feature Adapter to fix the Glioma/Meningioma boundary error.
"""

import json
from pathlib import Path

def create_notebook_e():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # --- INTRODUCTION ---
    add_md("""# Notebook E: Prototype-Anchored Contrastive Feature Adapter
**Scientific Protocol**
This notebook acts as the Next-Generation Architecture Implementation.
*Purpose*: Implement a targeted Supervised Contrastive Learning Adapter to mathematically separate the Glioma and Meningioma clusters, explicitly addressing the failure mode identified in Notebook C.
""")

    # --- PHASE 1: DATA INGESTION ---
    add_md("""## Phase 1: Secure Data Ingestion (Leakage Fix)
Load the pre-extracted frozen DINOv2 features and strictly isolate the test set from any adapter training or validation.""")
    
    add_code("""import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
except ImportError:
    os.system('pip install umap-learn')
    import umap

# Discover Latest Experiment Output
output_dirs = sorted(glob.glob("outputs/EXP_*"), reverse=True)
if not output_dirs:
    raise FileNotFoundError("No Notebook outputs found!")
EXP_DIR = Path(output_dirs[0])
print(f"Loading data from: {EXP_DIR}")

# Load Features
npz_data = np.load(EXP_DIR / 'features/dataset_features.npz', allow_pickle=True)
train_feat_all = npz_data['train_feat']
train_lbl_all = npz_data['train_lbl']
test_feat = npz_data['test_feat']
test_lbl = npz_data['test_lbl']

# Map string labels to integers if they aren't already
classes = sorted(np.unique(train_lbl_all))
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}

train_lbl_int = np.array([class_to_idx[l] for l in train_lbl_all])
test_lbl_int = np.array([class_to_idx[l] for l in test_lbl])

print(f"Total Training Samples: {len(train_feat_all)}")
print(f"Total Test Samples (Isolated): {len(test_feat)}")
print(f"Classes: {classes}")

# Create Train/Validation Split (85/15) to prevent overfitting
dataset = TensorDataset(torch.FloatTensor(train_feat_all), torch.LongTensor(train_lbl_int))
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"Adapter Train Split: {train_size} samples")
print(f"Adapter Val Split: {val_size} samples")
""")

    # --- PHASE 2: PYTORCH ARCHITECTURE ---
    add_md("""## Phase 2: PyTorch Projection Head Architecture
A lightweight MLP that projects the 768-D features into a 128-D L2-Normalized Metric Space.""")
    
    add_code("""class ContrastiveAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, out_dim=128):
        super(ContrastiveAdapter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        # Forward pass and L2 normalize
        x = self.net(x)
        x = F.normalize(x, p=2, dim=1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

adapter = ContrastiveAdapter().to(device)
print(adapter)
""")

    # --- PHASE 3: WEIGHTED SUPCON LOSS ---
    add_md("""## Phase 3: Custom Weighted Supervised Contrastive Loss
Implements SupCon with a temperature $\tau=0.1$ and a $2.0x$ penalty multiplier specifically for the `Glioma` and `Meningioma` boundary.""")
    
    add_code("""class WeightedSupConLoss(nn.Module):
    def __init__(self, temperature=0.1, boundary_multiplier=2.0):
        super(WeightedSupConLoss, self).__init__()
        self.temperature = temperature
        self.boundary_multiplier = boundary_multiplier
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # Calculate cosine similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-similarity from positive mask
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Create weights matrix to heavily penalize Glioma/Meningioma confusion
        weights = torch.ones_like(sim_matrix).to(device)
        glioma_idx = class_to_idx.get('glioma', -1)
        meningioma_idx = class_to_idx.get('meningioma', -1)
        
        if glioma_idx != -1 and meningioma_idx != -1:
            is_glioma = (labels == glioma_idx).float()
            is_mening = (labels == meningioma_idx).float()
            
            # Cross interactions: Glioma Anchor with Meningioma Negative
            glioma_mening_mask = torch.matmul(is_glioma, is_mening.T)
            mening_glioma_mask = torch.matmul(is_mening, is_glioma.T)
            
            # Apply multiplier
            weights += (glioma_mening_mask + mening_glioma_mask) * (self.boundary_multiplier - 1.0)
            
        # Compute log_prob
        exp_logits = torch.exp(sim_matrix) * logits_mask
        
        # Denominator incorporates the weights for negative pairs
        weighted_exp_logits = exp_logits * weights
        log_prob = sim_matrix - torch.log(weighted_exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        valid_anchors = mask_sum > 0
        
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)
            
        mean_log_prob_pos = (mask[valid_anchors] * log_prob[valid_anchors]).sum(1) / mask_sum[valid_anchors]
        
        loss = - mean_log_prob_pos
        return loss.mean()

criterion = WeightedSupConLoss(temperature=0.1, boundary_multiplier=2.0)
optimizer = optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
""")

    # --- PHASE 4: TRAINING LOOP ---
    add_md("""## Phase 4: Adapter Training Loop (Manifold Warping)
Train the projection head with Early Stopping based on the validation holdout.""")
    
    add_code("""epochs = 100
best_val_loss = float('inf')
patience = 15
patience_counter = 0

train_losses = []
val_losses = []

print("Starting Contrastive Adapter Training...")
for epoch in range(epochs):
    adapter.train()
    running_loss = 0.0
    for feats, lbls in train_loader:
        feats, lbls = feats.to(device), lbls.to(device)
        
        optimizer.zero_grad()
        embeddings = adapter(feats)
        loss = criterion(embeddings, lbls)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * feats.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation
    adapter.eval()
    val_loss = 0.0
    with torch.no_grad():
        for feats, lbls in val_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            embeddings = adapter(feats)
            loss = criterion(embeddings, lbls)
            val_loss += loss.item() * feats.size(0)
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(adapter.state_dict(), EXP_DIR / 'checkpoints/contrastive_adapter_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Plot training curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train SupCon Loss')
plt.plot(val_losses, label='Val SupCon Loss')
plt.title("Adapter Training Curve")
plt.legend()
plt.savefig(EXP_DIR / 'figures/adapter_training_curve.png')
plt.show()

# Load best weights
adapter.load_state_dict(torch.load(EXP_DIR / 'checkpoints/contrastive_adapter_best.pth'))
adapter.eval()
""")

    # --- PHASE 5: VISUAL & MATHEMATICAL PROOF ---
    add_md("""## Phase 5: Mathematical & Visual Proof of Separation
Extract the new 128-D features. Calculate the new Silhouette Score to mathematically prove the manifold is now separated into distinct clusters, and visualize with UMAP/PCA.""")
    
    add_code("""# Extract new features for ALL train and test
adapter.eval()
with torch.no_grad():
    train_feat_t = torch.FloatTensor(train_feat_all).to(device)
    test_feat_t = torch.FloatTensor(test_feat).to(device)
    
    new_train_feat = adapter(train_feat_t).cpu().numpy()
    new_test_feat = adapter(test_feat_t).cpu().numpy()

# Mathematical Proof: Silhouette Score
all_new_feat = np.vstack([new_train_feat, new_test_feat])
all_lbls = np.concatenate([train_lbl_int, test_lbl_int])

# Use subset if too large for quick silhouette
subset_size = min(5000, len(all_new_feat))
idx = np.random.choice(len(all_new_feat), subset_size, replace=False)
sil_score = silhouette_score(all_new_feat[idx], all_lbls[idx])

print(f"Old Notebook B2 Silhouette Score: ~0.14")
print(f"New Notebook E Silhouette Score: {sil_score:.4f}")

# Visual Proof: UMAP
print("Generating UMAP Visualization...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_emb = reducer.fit_transform(all_new_feat[idx])

plt.figure(figsize=(10, 8))
sns.scatterplot(x=umap_emb[:,0], y=umap_emb[:,1], hue=[idx_to_class[l] for l in all_lbls[idx]], palette='deep', s=10)
plt.title(f"UMAP of 128-D Adapted Features (Silhouette: {sil_score:.4f})")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig(EXP_DIR / 'figures/adapter_umap.png')
plt.show()
""")

    # --- PHASE 6: FINAL BENCHMARK ---
    add_md("""## Phase 6: The Final Benchmark (Breaking the 93.52% Ceiling)
We train a standard Logistic Regression on the new 128-D training features and evaluate on the 128-D test features, outputting the confusion matrix to prove the 57 Glioma $\to$ Meningioma errors are gone.""")
    
    add_code("""from sklearn.metrics import accuracy_score

print("Training Linear Probe on Adapted Features...")
clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(new_train_feat, train_lbl_int)

# Inference on isolated Test set
preds = clf.predict(new_test_feat)

acc = accuracy_score(test_lbl_int, preds)
print(f"\\n--- Notebook E Final Benchmark ---")
print(f"Test Accuracy: {acc*100:.2f}% (Baseline Notebook C Ceiling: 93.52%)")

print("\\nClassification Report:")
print(classification_report(test_lbl_int, preds, target_names=[idx_to_class[i] for i in range(4)]))

# Confusion Matrix
cm = confusion_matrix(test_lbl_int, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[idx_to_class[i] for i in range(4)], 
            yticklabels=[idx_to_class[i] for i in range(4)])
plt.title(f"Notebook E: Confusion Matrix\\nTest Acc: {acc*100:.2f}%")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig(EXP_DIR / 'figures/adapter_confusion_matrix.png')
plt.show()
""")

    output_path = "Notebook_E_Contrastive_Adapter.ipynb"
    with open(output_path, "w") as f:
        json.dump(nb, f, indent=4)
        
    print(f"Successfully generated {output_path}!")

if __name__ == "__main__":
    create_notebook_e()
