import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Set random seed for reproducibility
torch.manual_seed(42)

def main():
    print("Starting Improved Model Training (ResNet18)...")
    
    # 1. Define paths
    base_dir = '/Users/shashankchaurasia/Documents/tumor/dataset_archive2/archive'
    train_dir = os.path.join(base_dir, 'Training')
    test_dir = os.path.join(base_dir, 'Testing')
    
    # 2. Define Transformations with Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1822, 0.1822, 0.1822], std=[0.1783, 0.1783, 0.1783])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1822, 0.1822, 0.1822], std=[0.1783, 0.1783, 0.1783])
    ])
    
    # 3. Load Datasets
    print("Loading datasets...")
    # Load all training data with train transforms
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
    
    # Split Training into Train and Validation (80/20)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # We must ensure validation dataset uses val_test_transform (no augmentation)
    # We can override the transform for the validation split
    val_dataset.dataset = datasets.ImageFolder(train_dir, transform=val_test_transform)
    
    print(f"Data Splits: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 5. Define Model (ResNet18 Transfer Learning)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = len(full_train_dataset.classes)
    
    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler (reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # 6. Train the Model
    epochs = 20
    print(f"Training model for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs} | Time: {end_time - start_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
    # 7. Evaluate on Test Set
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    classes = full_train_dataset.classes
    print("\nClassification Report (Test Set):")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # 8. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - ResNet18')
    plt.savefig('improved_confusion_matrix.png')
    print("Saved confusion matrix to improved_confusion_matrix.png")

if __name__ == "__main__":
    main()
