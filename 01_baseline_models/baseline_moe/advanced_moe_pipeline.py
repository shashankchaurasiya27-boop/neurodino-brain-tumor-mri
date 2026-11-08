import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_clahe=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_clahe = is_clahe
        self.classes = sorted(os.listdir(root_dir))
        self.classes = [c for c in self.classes if not c.startswith('.')]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.startswith('.'):
                    continue
                self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.is_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def extract_vit_features(loader, model, device):
    model.eval()
    features = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # ViT forward returns the 768-dim embeddings since we set heads to Identity
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features), np.concatenate(labels_list)

def main():
    print("Initializing Advanced Mixture-of-Experts (MoE) Pipeline...")
    
    base_dir = '/Users/shashankchaurasia/Documents/tumor/dataset_archive2/archive'
    train_dir = os.path.join(base_dir, 'Training')
    test_dir = os.path.join(base_dir, 'Testing')
    
    # ViT expects 224x224 and standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading and applying CLAHE to MRI scans...")
    full_train_dataset = MRIDataset(train_dir, transform=transform, is_clahe=True)
    test_dataset = MRIDataset(test_dir, transform=transform, is_clahe=True)
    
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup ViT Feature Extractor
    print("Loading pre-trained ViT_B_16...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    vit.heads = nn.Identity() # Remove classification head to get 768-dim embeddings
    vit = vit.to(device)
    
    print("Extracting Deep Features (ViT Embeddings) for Training Set...")
    start_time = time.time()
    X_train, y_train = extract_vit_features(train_loader, vit, device)
    print(f"Extracted Train Features: {X_train.shape} in {time.time()-start_time:.1f}s")
    
    print("Extracting Features for Validation Set...")
    X_val, y_val = extract_vit_features(val_loader, vit, device)
    print("Extracting Features for Testing Set...")
    X_test, y_test = extract_vit_features(test_loader, vit, device)
    
    # Unsupervised Clustering (Orientation Discovery)
    print("\nRunning K-Means (K=3) to automatically discover anatomical orientations...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    train_clusters = kmeans.fit_predict(X_train)
    val_clusters = kmeans.predict(X_val)
    test_clusters = kmeans.predict(X_test)
    
    cluster_counts = Counter(train_clusters)
    print(f"Cluster sizes: {cluster_counts}")
    
    # Train Mixture of Experts
    print("\nTraining 3 Expert PyTorch Linear Classifiers (One per cluster)...")
    experts = []
    expert_losses = {0: [], 1: [], 2: []}
    for k in range(3):
        # Subset data for this expert
        idx_k = (train_clusters == k)
        X_k = torch.FloatTensor(X_train[idx_k]).to(device)
        y_k = torch.LongTensor(y_train[idx_k]).to(device)
        
        # Calculate class weights for this specific cluster
        c_counts = np.bincount(y_train[idx_k], minlength=4)
        # Avoid division by zero if a cluster lacks a class
        weights = np.zeros(4)
        for i in range(4):
            if c_counts[i] > 0:
                weights[i] = 1.0 / c_counts[i]
        weights = weights / weights.sum()
        
        classifier = nn.Sequential(nn.Linear(768, 4)).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
        optimizer = optim.Adam(classifier.parameters(), lr=0.01)
        
        epochs = 50
        print(f"\n--- Training Expert {k} (Samples: {sum(idx_k)}) ---")
        for epoch in range(epochs):
            classifier.train()
            optimizer.zero_grad()
            outputs = classifier(X_k)
            loss = criterion(outputs, y_k)
            loss.backward()
            optimizer.step()
            expert_losses[k].append(loss.item())
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
        classifier.eval()
        experts.append(classifier)
        
    # MoE Inference (Routing)
    print("\nPerforming MoE Inference (Routing to Experts)...")
    
    def evaluate_moe(X, y, clusters):
        X_t = torch.FloatTensor(X).to(device)
        all_probs = np.zeros((len(X), 4))
        
        with torch.no_grad():
            for k in range(3):
                idx_k = (clusters == k)
                if sum(idx_k) == 0: continue
                # Route to Expert K
                expert = experts[k]
                out_k = expert(X_t[idx_k])
                probs_k = torch.softmax(out_k, dim=1).cpu().numpy()
                all_probs[idx_k] = probs_k
                
        preds = np.argmax(all_probs, axis=1)
        return preds, all_probs
        
    val_preds, _ = evaluate_moe(X_val, y_val, val_clusters)
    val_acc = (val_preds == y_val).mean()
    print(f"Validation Accuracy (MoE): {val_acc*100:.2f}%")
    
    test_preds, test_probs = evaluate_moe(X_test, y_test, test_clusters)
    
    classes = full_train_dataset.classes
    print("\n--- Final Test Set Evaluation (MoE) ---")
    print(classification_report(y_test, test_preds, target_names=classes))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Advanced MoE Pipeline (ViT + KMeans)')
    plt.tight_layout()
    plt.savefig('moe_confusion_matrix.png')
    
    # Generate ROC Curves
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - MoE Pipeline')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('moe_roc_curves.png')
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i in range(4):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], test_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'{classes[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - MoE Pipeline')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('moe_pr_curves.png')

    # 4. Expert Learning Curves
    plt.figure(figsize=(8, 6))
    for k in range(3):
        plt.plot(expert_losses[k], label=f'Expert {k} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss per Expert')
    plt.legend()
    plt.tight_layout()
    plt.savefig('moe_expert_learning_curves.png')

    # 5. PCA Visualization (Clusters & True Classes)
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter1 = ax1.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters, cmap='viridis', alpha=0.6)
    ax1.set_title('PCA of ViT Features (Colored by Discovered Orientation)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)

    scatter2 = ax2.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='tab10', alpha=0.6)
    ax2.set_title('PCA of ViT Features (Colored by True Tumor Class)')
    ax2.set_xlabel('Principal Component 1')
    legend2 = ax2.legend(handles=scatter2.legend_elements()[0], labels=classes, title="Classes")
    ax2.add_artist(legend2)

    plt.tight_layout()
    plt.savefig('moe_pca_visualization.png')
    
    print("Saved all possible MoE evaluation graphs!")

if __name__ == "__main__":
    main()
