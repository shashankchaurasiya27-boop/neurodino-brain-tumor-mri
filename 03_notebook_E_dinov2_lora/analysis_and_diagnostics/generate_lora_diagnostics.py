#!/usr/bin/env python3
"""
Notebook F / Intervention B: Comprehensive LoRA Diagnostics & Logging
================================================================================
This script takes the trained LoRA DINOv2 model and generates exhaustive
scientific graphs (ROC, PR Curves, UMAP) and saves all raw probabilities/logs
for future research.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    os.system('pip install peft')
    from peft import LoraConfig, get_peft_model

try:
    import umap
except ImportError:
    os.system('pip install umap-learn')
    import umap

def main():
    print("--- Phase 1: Environment & Data Preparation ---")
    DATASET_PATH = Path("../dataset_archive2/archive")
    if not DATASET_PATH.exists():
        DATASET_PATH = Path("dataset_archive2/archive")
        
    TEST_DIR = DATASET_PATH / "Testing"
    EXP_DIR = Path("outputs/EXP_LORA")
    os.makedirs(EXP_DIR / "logs", exist_ok=True)
    os.makedirs(EXP_DIR / "figures", exist_ok=True)
    os.makedirs(EXP_DIR / "features", exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Testing Samples: {len(test_dataset)}")

    print("\n--- Phase 2: Restoring the LoRA Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
    )
    dinov2_lora = get_peft_model(dinov2, lora_config)

    class DINOv2Classifier(nn.Module):
        def __init__(self, backbone, num_classes=4):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.BatchNorm1d(768),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features), features

    model = DINOv2Classifier(dinov2_lora, num_classes=num_classes).to(device)
    
    checkpoint_path = EXP_DIR / "checkpoints/lora_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}! Is the training script finished?")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("LoRA Model Successfully Restored.")

    print("\n--- Phase 3: Exhaustive Inference & Raw Logging ---")
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    file_paths = [s[0] for s in test_dataset.samples]

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, features = model(inputs)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)

    # Save raw features and labels for future clustering experiments
    np.savez(EXP_DIR / "features/lora_test_features.npz", features=all_features, labels=all_labels)
    
    # Save raw CSV log of probabilities for future result generation
    log_df = pd.DataFrame({
        "filepath": file_paths,
        "true_label": [class_names[i] for i in all_labels],
        "predicted_label": [class_names[i] for i in all_preds],
        "prob_glioma": all_probs[:, class_names.index('glioma')],
        "prob_meningioma": all_probs[:, class_names.index('meningioma')],
        "prob_notumor": all_probs[:, class_names.index('notumor')],
        "prob_pituitary": all_probs[:, class_names.index('pituitary')]
    })
    csv_path = EXP_DIR / "logs/lora_test_predictions_log.csv"
    log_df.to_csv(csv_path, index=False)
    print(f"Saved massive RAW PROBABILITY log to: {csv_path}")

    print("\n--- Phase 4: Generating Exhaustive Graphs ---")
    
    # 1. Classification Report & Confusion Matrix
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title("LoRA DINOv2: Detailed Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(EXP_DIR / "figures/lora_detailed_confusion_matrix.png")
    
    # 2. ROC Curves (One-vs-Rest)
    labels_bin = label_binarize(all_labels, classes=[0,1,2,3])
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']
    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - LoRA DINOv2')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / "figures/lora_roc_curves.png")
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(num_classes), colors):
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f'{class_names[i]}')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - LoRA DINOv2')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / "figures/lora_pr_curves.png")
    
    # 4. UMAP Projection of the LoRA Features
    print("Generating UMAP Projection of LoRA Learned Features...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_emb[:,0], y=umap_emb[:,1], hue=[class_names[l] for l in all_labels], palette='deep', s=15)
    plt.title(f"UMAP Projection of LoRA Fine-Tuned DINOv2 Features")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(EXP_DIR / "figures/lora_umap.png")
    
    print("\n--- ALL LOGS AND GRAPHS SAVED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()
