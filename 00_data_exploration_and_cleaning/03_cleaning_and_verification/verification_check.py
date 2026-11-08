import os
import glob
import hashlib
import pandas as pd
from itertools import combinations
import cv2
import numpy as np

dataset_dir = '/Users/shashankchaurasia/Downloads/tumor/dataset_archive2/archive'

def get_image_info(path):
    with open(path, 'rb') as f:
        data = f.read()
        md5 = hashlib.md5(data).hexdigest()
    
    split = "Training" if "Training" in path else "Testing"
    return {'path': path, 'md5': md5, 'split': split}

print("Scanning cleaned dataset for remaining duplicates...")

# Gather all images in Training and Testing, excluding the 'duplicates' folder
all_files = glob.glob(os.path.join(dataset_dir, '**', '*.*'), recursive=True)
valid_files = [f for f in all_files if 'duplicates' not in f and os.path.isfile(f) and f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"Found {len(valid_files)} images in the cleaned dataset.")

data = [get_image_info(f) for f in valid_files]
df = pd.DataFrame(data)

# Check for Exact Duplicates (MD5 matches)
md5_counts = df['md5'].value_counts()
dup_md5s = md5_counts[md5_counts > 1].index

exact_dups = []
cross_split_leakage = 0
intra_split_dups = 0

for md5 in dup_md5s:
    group = df[df['md5'] == md5]
    splits = group['split'].unique()
    
    if len(splits) > 1:
        cross_split_leakage += 1
    else:
        intra_split_dups += 1
        
    exact_dups.append(group['path'].tolist())

# Check for Byte-Different Pixel Identical (we can just load images and hash their pixel arrays)
print("Checking for pixel-identical images (ignoring metadata)...")
pixel_hashes = {}
pixel_dups = 0
for f in valid_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Hash the raw pixel bytes
        ph = hashlib.md5(img.tobytes()).hexdigest()
        if ph in pixel_hashes:
            pixel_dups += 1
        else:
            pixel_hashes[ph] = f

report = f"""# Post-Cleaning Verification Report

## 1. Dataset Status
- **Images Scanned:** {len(valid_files)}
- **Directories Scanned:** `Training/` and `Testing/` (excluding `duplicates/`)

## 2. Duplicate Findings
- **Remaining Exact MD5 Duplicates:** {len(exact_dups)} groups found
- **Remaining Cross-Split Exact Leakage:** {cross_split_leakage} instances
- **Remaining Intra-Split Exact Duplicates:** {intra_split_dups} instances
- **Remaining Pixel-Identical (Byte-Different) Duplicates:** {pixel_dups} instances

## 3. Verdict
"""
if len(exact_dups) == 0 and pixel_dups == 0:
    report += "✅ **PASSED**: The dataset is completely clean. Absolutely ZERO exact duplicates, leakage, or pixel-identical images remain."
else:
    report += "❌ **FAILED**: Remaining duplicates were detected. Further cleaning required."

report_path = os.path.join(dataset_dir, 'post_cleaning_verification.md')
with open(report_path, 'w') as f:
    f.write(report)
    
print("Verification complete! See post_cleaning_verification.md")
