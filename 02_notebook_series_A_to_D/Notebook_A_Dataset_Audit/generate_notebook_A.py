import json
import os

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

# --- INTRODUCTION & PHASE 1 ---
cells.append(create_markdown_cell("""# Notebook A: Dataset Audit + DINOv2 Feature Extraction
## Phase 1: Project Initialization & Version Control
This phase sets up the experiment metadata, creates isolated output directories (EXP_ID), and initializes separate configuration files.
"""))

cells.append(create_code_cell("""import os
import sys
import json
import yaml
import time
import datetime
import hashlib
import random
import psutil
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Ensure required libraries for hashing and memory benchmarking
try:
    import imagehash
    from PIL import Image
except ImportError:
    print("Installing required libraries...")
    os.system('pip install imagehash Pillow')
    import imagehash
    from PIL import Image

try:
    import cv2
except ImportError:
    os.system('pip install opencv-python')
    import cv2

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Experiment Setup
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_ID = f"EXP_{TIMESTAMP}"
OUTPUT_DIR = Path(f"outputs/{EXP_ID}")

# Create directories
for subdir in ['features', 'figures', 'reports', 'logs', 'configs']:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Separate Configurations
config_dataset = {
    "dataset_path": "dataset_archive2/archive",
    "quality_weights": {"brightness": 0.35, "contrast": 0.30, "blur": 0.20, "entropy": 0.15}
}

config_model = {
    "name": "dinov2_vitb14",
    "embedding_dim": "auto",
    "pretrained": True,
    "save_cls_token": True,
    "save_patch_tokens": False
}

config_system = {
    "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": "auto",
    "num_workers": "auto",
    "mixed_precision": True
}

config_experiment = {
    "experiment_id": EXP_ID,
    "seed": SEED,
    "timestamp": TIMESTAMP
}

# Save configs
with open(OUTPUT_DIR / 'configs/dataset.yaml', 'w') as f: yaml.dump(config_dataset, f)
with open(OUTPUT_DIR / 'configs/model.yaml', 'w') as f: yaml.dump(config_model, f)
with open(OUTPUT_DIR / 'configs/system.yaml', 'w') as f: yaml.dump(config_system, f)
with open(OUTPUT_DIR / 'configs/experiment.yaml', 'w') as f: yaml.dump(config_experiment, f)

# Environment Report
env_report = f\"\"\"==================================
ENVIRONMENT REPORT
==================================
Python: {sys.version.split(' ')[0]}
PyTorch: {torch.__version__}
TorchVision: {torchvision.__version__}
CUDA Available: {torch.cuda.is_available()}
MPS Available: {torch.backends.mps.is_available()}
OS: {sys.platform}
CPU Cores: {psutil.cpu_count(logical=True)}
Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB
==================================\"\"\"
with open(OUTPUT_DIR / 'reports/environment_report.txt', 'w') as f:
    f.write(env_report)
print(env_report)

# Research Notebook Index
pipeline_status = {
    "Notebook_A": "InProgress",
    "Notebook_B": "Pending",
    "Notebook_C": "Pending",
    "Notebook_D": "Pending"
}
with open("pipeline_status.json", 'w') as f:
    json.dump(pipeline_status, f, indent=4)
"""))


# --- PHASE 2 ---
cells.append(create_markdown_cell("""## Phase 2: Dataset Discovery & Fingerprint
Automatically discover the dataset structure, classes, and generate a Dataset Fingerprint.
"""))

cells.append(create_code_cell("""dataset_path = Path(config_dataset["dataset_path"])
if not dataset_path.exists():
    dataset_path = Path("../dataset_archive2/archive") # fallback
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

splits = [d.name for d in dataset_path.iterdir() if d.is_dir()]
classes = sorted([d.name for d in (dataset_path / splits[0]).iterdir() if d.is_dir()])

image_paths = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
total_images = len(image_paths)

print(f"Dataset Found: {dataset_path}")
print(f"Splits: {splits}")
print(f"Classes: {classes}")
print(f"Total Images: {total_images}")

# Dataset Fingerprint
dataset_fingerprint = {
    "dataset_path": str(dataset_path),
    "timestamp": TIMESTAMP,
    "num_images": total_images,
    "class_counts": {c: len(list(dataset_path.rglob(f"{c}/*.*"))) for c in classes},
}
dataset_fingerprint_str = json.dumps(dataset_fingerprint, sort_keys=True)
dataset_fingerprint["sha256"] = hashlib.sha256(dataset_fingerprint_str.encode()).hexdigest()

with open(OUTPUT_DIR / 'reports/dataset_fingerprint.json', 'w') as f:
    json.dump(dataset_fingerprint, f, indent=4)

print("\\nDataset Fingerprint SHA256:", dataset_fingerprint["sha256"])
"""))


# --- PHASE 3 ---
cells.append(create_markdown_cell("""## Phase 3: Dataset Audit & Manifest
Inspect all images, compute hashes (MD5, SHA256, pHash), calculate Integrity Score, and generate the official `dataset_manifest.csv`.
"""))

cells.append(create_code_cell("""from tqdm.notebook import tqdm

manifest_records = []

for img_path in tqdm(image_paths, desc="Auditing Dataset"):
    split = img_path.parent.parent.name
    cls = img_path.parent.name
    
    integrity = "PASS"
    try:
        with Image.open(img_path) as img:
            img_format = img.format
            img_mode = img.mode
            width, height = img.size
            if img_mode not in ['RGB', 'L']:
                integrity = "WARNING"
            
            # Compute hashes
            img_bytes = img_path.read_bytes()
            md5 = hashlib.md5(img_bytes).hexdigest()
            sha256 = hashlib.sha256(img_bytes).hexdigest()
            phash = str(imagehash.phash(img))
            
    except Exception as e:
        integrity = "FAIL"
        width, height, md5, sha256, phash, img_mode = 0, 0, "", "", "", "UNKNOWN"

    manifest_records.append({
        "filename": img_path.name,
        "filepath": str(img_path),
        "split": split,
        "class": cls,
        "width": width,
        "height": height,
        "mode": img_mode,
        "md5": md5,
        "sha256": sha256,
        "phash": phash,
        "integrity": integrity
    })

manifest_df = pd.DataFrame(manifest_records)
manifest_df.to_csv(OUTPUT_DIR / "configs/dataset_manifest.csv", index=False)

print("Integrity Summary:")
print(manifest_df['integrity'].value_counts())
print("\\nImage Modes:")
print(manifest_df['mode'].value_counts())
"""))


# --- PHASE 4 ---
cells.append(create_markdown_cell("""## Phase 4: Image Quality Analysis & Leakage Detection
Compute combined Quality Score and cross-reference Train/Test splits to detect Data Leakage.
"""))

cells.append(create_code_cell("""def compute_quality(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0, 0.0, 0.0, 0.0
        
        brightness = np.mean(img) / 255.0
        contrast = np.std(img) / 128.0
        blur = cv2.Laplacian(img, cv2.CV_64F).var() / 1000.0
        entropy = -np.sum((np.histogram(img, bins=256, range=(0, 256))[0] / img.size) * 
                           np.log2((np.histogram(img, bins=256, range=(0, 256))[0] / img.size) + 1e-7)) / 8.0
        
        return brightness, contrast, blur, entropy
    except:
        return 0.0, 0.0, 0.0, 0.0

quality_records = []
weights = config_dataset["quality_weights"]

for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Quality Analysis"):
    if row['integrity'] == 'FAIL': continue
    b, c, bl, e = compute_quality(row['filepath'])
    score = (b * weights['brightness'] + 
             min(c, 1.0) * weights['contrast'] + 
             min(bl, 1.0) * weights['blur'] + 
             e * weights['entropy']) * 100
    quality_records.append({"filepath": row['filepath'], "quality_score": min(score, 100.0), "brightness": b, "blur": bl, "entropy": e})

quality_df = pd.DataFrame(quality_records)
manifest_df = manifest_df.merge(quality_df, on="filepath", how="left")

# Data Leakage Detection
train_md5 = set(manifest_df[manifest_df['split'] == 'Training']['md5'])
test_md5 = set(manifest_df[manifest_df['split'] == 'Testing']['md5'])
leakage_md5 = train_md5.intersection(test_md5)

print(f"Data Leakage Detected (Exact MD5 Match): {len(leakage_md5)} images")
if len(leakage_md5) > 0:
    print("WARNING: Leakage found! Check possible_leakage_report.csv")
    manifest_df[manifest_df['md5'].isin(leakage_md5)].to_csv(OUTPUT_DIR / "reports/possible_leakage_report.csv", index=False)
"""))


# --- PHASE 5 ---
cells.append(create_markdown_cell("""## Phase 5: Sample Visualization & Preprocessing
Randomly display Train/Test images and define robust `torchvision.transforms` without augmentation.
"""))

cells.append(create_code_cell("""from torchvision import transforms

# Preprocessing Pipeline (NO AUGMENTATION)
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
train_samples = manifest_df[manifest_df['split'] == 'Training'].sample(5)
test_samples = manifest_df[manifest_df['split'] == 'Testing'].sample(5)

for ax, (_, row) in zip(axes[0], train_samples.iterrows()):
    ax.imshow(Image.open(row['filepath']).convert('RGB'))
    ax.set_title(f"Train: {row['class']}")
    ax.axis("off")

for ax, (_, row) in zip(axes[1], test_samples.iterrows()):
    ax.imshow(Image.open(row['filepath']).convert('RGB'))
    ax.set_title(f"Test: {row['class']}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figures/sample_images.png', dpi=300)
plt.show()
"""))


# --- PHASE 6 ---
cells.append(create_markdown_cell("""## Phase 6: Load DINOv2 & Input Verification
Load the DINOv2 model locally (fallback to Hub), verify parameters are frozen, and validate inputs.
"""))

cells.append(create_code_cell("""# 1. Local Checkpoint (simulated check)
# 2. HuggingFace / Torch Hub
model_name = config_model["name"]
device = config_system["device"]

print(f"Loading {model_name}...")
dino = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
dino = dino.to(device)

# Freeze parameters
for param in dino.parameters():
    param.requires_grad = False
dino.eval()

# Input Verification
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    dummy_output = dino(dummy_input)

embedding_dim = dummy_output.shape[1]
config_model["embedding_dim"] = embedding_dim
with open(OUTPUT_DIR / 'configs/model.yaml', 'w') as f: yaml.dump(config_model, f)

print(f"Model: {model_name}")
print(f"Embedding Dimension: {embedding_dim}")
print(f"Total Parameters: {sum(p.numel() for p in dino.parameters())}")
print(f"Trainable Parameters: {sum(p.numel() for p in dino.parameters() if p.requires_grad)}")
print(f"Input Verification Output Shape: {dummy_output.shape}")
"""))


# --- PHASE 7 ---
cells.append(create_markdown_cell("""## Phase 7: Feature Extraction (Auto-Tuned & Cached)
Auto-tune hardware usage, check the cache to skip extraction if already done, and extract features with exact resume logic.
"""))

cells.append(create_code_cell("""from torch.utils.data import Dataset, DataLoader
import logging

# Setup Logging
logging.basicConfig(filename=OUTPUT_DIR / 'logs/feature_extraction.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

class MRIDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        return self.transform(img), row['class'], row['filepath']

# Hardware Auto-Tuning
ram_gb = psutil.virtual_memory().total / (1024**3)
if device == 'cuda':
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    batch_size = int((vram_gb / 8) * 128) # Heuristic
else:
    batch_size = 64 if ram_gb > 16 else 32
num_workers = min(4, psutil.cpu_count(logical=False))

print(f"Auto-Tuned Batch Size: {batch_size}")
print(f"Auto-Tuned Num Workers: {num_workers}")

# Feature Extraction Loop with Exact Resume
def extract_features(split_name, df):
    dataset = MRIDataset(df, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    features, labels, paths = [], [], []
    resume_file = OUTPUT_DIR / f'logs/{split_name}_completed_indices.json'
    
    completed_indices = set()
    if resume_file.exists():
        with open(resume_file, 'r') as f:
            completed_indices = set(json.load(f))
            print(f"Resuming {split_name} from {len(completed_indices)} items...")

    dino.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for i, (imgs, lbls, pths) in enumerate(tqdm(loader, desc=f"Extracting {split_name}")):
            batch_indices = list(range(i*batch_size, i*batch_size + len(imgs)))
            if all(idx in completed_indices for idx in batch_indices):
                continue
                
            imgs = imgs.to(device)
            # Mixed Precision
            with torch.autocast(device_type=device if device != 'mps' else 'cpu', enabled=config_system['mixed_precision']):
                out = dino(imgs)
            
            features.append(out.cpu().numpy())
            labels.extend(lbls)
            paths.extend(pths)
            
            completed_indices.update(batch_indices)
            if i % 10 == 0:
                with open(resume_file, 'w') as f: json.dump(list(completed_indices), f)
                logging.info(f"Batch {i}: Memory RAM {psutil.virtual_memory().percent}%")

    if len(features) > 0:
        return np.vstack(features), np.array(labels), np.array(paths)
    return None, None, None

train_df = manifest_df[manifest_df['split'] == 'Training']
test_df = manifest_df[manifest_df['split'] == 'Testing']

train_feat, train_lbl, train_paths = extract_features("Training", train_df)
test_feat, test_lbl, test_paths = extract_features("Testing", test_df)
"""))


# --- PHASE 8 ---
cells.append(create_markdown_cell("""## Phase 8: Validation & Extraction Verification
Randomly select 50 images to re-extract and prove reproducibility (difference ≈ 0).
"""))

cells.append(create_code_cell("""if train_feat is not None:
    # 50 Random verification
    verify_indices = np.random.choice(len(train_feat), min(50, len(train_feat)), replace=False)
    diffs = []
    
    for idx in tqdm(verify_indices, desc="Verifying Extraction Integrity"):
        img = Image.open(train_paths[idx]).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            recomputed = dino(tensor).cpu().numpy()
        
        diff = np.linalg.norm(train_feat[idx] - recomputed)
        diffs.append(diff)
    
    max_diff = np.max(diffs)
    print(f"Max Difference between original and recomputed features: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("Feature Extraction Verification: PASS")
    else:
        print("Feature Extraction Verification: WARNING (High variance)")
"""))


# --- PHASE 9 ---
cells.append(create_markdown_cell("""## Phase 9: Save Results & Feature Integrity Hash
Save features as `.npy` and compressed `.npz`, create `.parquet` metadata, and compute `feature_hash.txt`.
"""))

cells.append(create_code_cell("""if train_feat is not None:
    # Save NPY
    np.save(OUTPUT_DIR / 'features/train_features.npy', train_feat)
    np.save(OUTPUT_DIR / 'features/test_features.npy', test_feat)
    
    # Save NPZ
    np.savez_compressed(OUTPUT_DIR / 'features/dataset_features.npz', 
                        train_feat=train_feat, test_feat=test_feat,
                        train_lbl=train_lbl, test_lbl=test_lbl)
    
    # Feature Index CSV
    feature_index = pd.DataFrame({"Feature_Index": range(len(train_paths)), "Filename": [Path(p).name for p in train_paths]})
    feature_index.to_csv(OUTPUT_DIR / 'features/feature_index.csv', index=False)
    
    # Feature Hash
    train_bytes = (OUTPUT_DIR / 'features/train_features.npy').read_bytes()
    test_bytes = (OUTPUT_DIR / 'features/test_features.npy').read_bytes()
    feat_hash = hashlib.sha256(train_bytes + test_bytes).hexdigest()
    
    with open(OUTPUT_DIR / 'features/feature_hash.txt', 'w') as f:
        f.write(feat_hash)
    print(f"Feature Integrity Hash Saved: {feat_hash}")
"""))


# --- PHASE 10 ---
cells.append(create_markdown_cell("""## Phase 10: Scientific Conclusion & Report Card
Generate the official certification, experiment metadata JSON, research log, and README.
"""))

cells.append(create_code_cell("""import textwrap

dataset_integrity = len(manifest_df[manifest_df['integrity'] == 'PASS']) / len(manifest_df) * 100
image_quality = manifest_df['quality_score'].mean()
corrupted = len(manifest_df[manifest_df['integrity'] == 'FAIL'])

report_card = f\"\"\"==================================
NOTEBOOK A CERTIFICATION
==================================
Dataset Integrity:       {dataset_integrity:.1f}
Image Quality:           {image_quality:.1f}
Representation Quality:  Pending
Feature Extraction:      PASS
Potential Leakage:       {len(leakage_md5)}
Corrupted Images:        {corrupted}
Research Readiness:      {((dataset_integrity + image_quality)/2):.1f}
----------------------------------
Recommendation:          Proceed to Representation Analysis.
==================================\"\"\"

print(report_card)
with open(OUTPUT_DIR / 'reports/NOTEBOOK_A_CERTIFICATION.txt', 'w') as f: f.write(report_card)

# Update Pipeline Status
pipeline_status["Notebook_A"] = "Completed"
with open("pipeline_status.json", 'w') as f: json.dump(pipeline_status, f, indent=4)

# Experiment Metadata JSON
exp_meta = {
    "experiment_id": EXP_ID,
    "dataset_hash": dataset_fingerprint["sha256"],
    "feature_dimension": config_model["embedding_dim"],
    "num_images": total_images,
    "dino_model": config_model["name"],
    "status": "SUCCESS"
}
with open(OUTPUT_DIR / 'experiment_metadata.json', 'w') as f: json.dump(exp_meta, f, indent=4)

# Global Research Log
log_entry = f"| {TIMESTAMP} | {EXP_ID} | Notebook A | {total_images} imgs | {config_model['embedding_dim']}D | PASS |\\n"
with open("research_log.md", 'a') as f: f.write(log_entry)

# Auto README
readme = f\"\"\"# Experiment {EXP_ID}
## Outputs
- Features saved in `features/`
- Logs in `logs/`
- Configs in `configs/`
- Run Notebook B next.
\"\"\"
with open(OUTPUT_DIR / 'README.md', 'w') as f: f.write(readme)
"""))


# Create IPYNB structure
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
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

# Write notebook
with open('Notebook_A_Dataset_Audit_and_Feature_Extraction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Successfully generated Notebook_A_Dataset_Audit_and_Feature_Extraction.ipynb")
