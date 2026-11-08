import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.is_clahe:
            # Apply Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
        # Convert grayscale back to RGB so ResNet can accept it
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def extract_features(loader, model, device):
    model.eval()
    features = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Forward pass through ResNet50 (without final FC layer)
            outputs = model(inputs)
            # Flatten spatial dimensions if needed (ResNet outputs Bx2048x1x1)
            outputs = outputs.view(outputs.size(0), -1) 
            
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features), np.concatenate(labels_list)

def main():
    print("Initializing Clinical MRI Pipeline...")
    
    base_dir = '/Users/shashankchaurasia/Downloads/tumor/dataset_archive2/archive'
    train_dir = os.path.join(base_dir, 'Training')
    test_dir = os.path.join(base_dir, 'Testing')
    
    # Standard ResNet preprocessing. Since we use CLAHE, we don't need dataset-specific mean/std
    # because CLAHE normalizes the intensity spread. We'll use standard ImageNet values for the ResNet50 weights.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Load Datasets
    print("Loading and applying CLAHE to MRI scans...")
    full_train_dataset = MRIDataset(train_dir, transform=transform, is_clahe=True)
    test_dataset = MRIDataset(test_dir, transform=transform, is_clahe=True)
    
    # Split Training into Train and Validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    print(f"Data Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 2. Setup Feature Extractor (ResNet50)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ResNet50 and remove the final classification layer to get 2048-dim embeddings
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1] # Remove last FC layer
    model = nn.Sequential(*modules)
    model = model.to(device)
    
    # 3. Extract Deep Features
    print("Extracting Deep Features (ResNet50 Embeddings) for Training Set...")
    start_time = time.time()
    X_train, y_train = extract_features(train_loader, model, device)
    print(f"Extracted Train Features: {X_train.shape} in {time.time()-start_time:.1f}s")
    
    print("Extracting Features for Validation Set...")
    X_val, y_val = extract_features(val_loader, model, device)
    
    print("Extracting Features for Testing Set...")
    X_test, y_test = extract_features(test_loader, model, device)
    
    # 4. Train Fast PyTorch Linear Classifier on Deep Features
    print("\nTraining PyTorch Linear Classifier on Deep Features for learning curves...")
    
    # Convert numpy arrays to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Define Linear Model (Equivalent to Logistic Regression)
    classifier = nn.Sequential(
        nn.Linear(2048, 4)
    ).to(device)
    
    # Calculate class weights for imbalance
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    weights = weights / weights.sum()
    class_weights = torch.FloatTensor(weights).to(device)
    
    criterion_clf = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_clf = optim.Adam(classifier.parameters(), lr=0.01)
    
    epochs = 50
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    clf_start = time.time()
    for epoch in range(epochs):
        classifier.train()
        optimizer_clf.zero_grad()
        
        outputs = classifier(X_train_t)
        loss = criterion_clf(outputs, y_train_t)
        loss.backward()
        optimizer_clf.step()
        
        train_losses.append(loss.item())
        _, preds = torch.max(outputs, 1)
        train_acc = (preds == y_train_t).float().mean().item()
        train_accs.append(train_acc)
        
        # Validation
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(X_val_t)
            v_loss = criterion_clf(val_outputs, y_val_t)
            val_losses.append(v_loss.item())
            _, v_preds = torch.max(val_outputs, 1)
            val_acc = (v_preds == y_val_t).float().mean().item()
            val_accs.append(val_acc)
            
    print(f"Classifier training ({epochs} epochs) completed in {time.time()-clf_start:.3f}s")
    
    # 5. Generate Training Curves (Loss & Accuracy)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Saved training curves to 'training_curves.png'")
    
    # 6. Evaluate on Test Set
    print("\n--- Final Test Set Evaluation ---")
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(X_test_t)
        test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
        _, test_preds_t = torch.max(test_outputs, 1)
        test_preds = test_preds_t.cpu().numpy()
        
    classes = full_train_dataset.classes
    print(classification_report(y_test, test_preds, target_names=classes))
    
    # 7. Generate Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Clinical Pipeline')
    plt.tight_layout()
    plt.savefig('clinical_pipeline_confusion_matrix.png')
    
    # 8. Generate ROC Curves (Medical Imaging Standard)
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
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
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    print("Saved ROC curves to 'roc_curves.png'")

if __name__ == "__main__":
    main()
