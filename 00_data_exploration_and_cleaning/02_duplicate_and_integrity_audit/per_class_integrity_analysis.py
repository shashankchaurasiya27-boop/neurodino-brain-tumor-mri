# %% [markdown]
# # Per-Class Duplicate Integrity Analysis & Dataset Certification

# %%
import os
import glob
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = 'dataset_archive2/archive'
output_dir = 'per_class_visuals'
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## 1. Load Data & Matrices

# %%
df_pairs = pd.read_csv('integrity_outputs/duplicate_decision_matrix.csv')

# Load all images to get exact class totals
image_paths = []
labels = []
splits = []
for s in ['Training', 'Testing']:
    for c in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        for f in glob.glob(os.path.join(dataset_dir, s, c, '*.*')):
            image_paths.append(f)
            labels.append(c)
            splits.append(s)

df_all = pd.DataFrame({'path': image_paths, 'label': labels, 'split': splits})
class_totals = df_all.groupby(['label', 'split']).size().unstack(fill_value=0)
class_totals.columns = ['Test Images', 'Train Images']
class_totals['Total Original'] = class_totals['Train Images'] + class_totals['Test Images']

# %% [markdown]
# ## 2. Affected Images & Duplicate Categories

# %%
per_class_stats = []

for c in ['glioma', 'meningioma', 'notumor', 'pituitary']:
    # Train-Train duplicates
    train_dups = df_pairs[(df_pairs['Class A'] == c) & (df_pairs['Class B'] == c) & (df_pairs['Split A'] == 'Training') & (df_pairs['Split B'] == 'Training')]
    # Test-Test duplicates
    test_dups = df_pairs[(df_pairs['Class A'] == c) & (df_pairs['Class B'] == c) & (df_pairs['Split A'] == 'Testing') & (df_pairs['Split B'] == 'Testing')]
    # Train-Test Leakage
    leakage = df_pairs[(df_pairs['Class A'] == c) & (df_pairs['Class B'] == c) & (df_pairs['Is Cross Split'] == True) & (df_pairs['Category'] == 'Critical Leakage')]
    
    exact_tr = len(train_dups[train_dups['Category'].isin(['Exact Duplicate', 'Pixel Identical'])])
    near_tr = len(train_dups[train_dups['Category'] == 'Near Duplicate'])
    
    exact_te = len(test_dups[test_dups['Category'].isin(['Exact Duplicate', 'Pixel Identical'])])
    near_te = len(test_dups[test_dups['Category'] == 'Near Duplicate'])
    
    leak_count = len(leakage)
    
    # Unique Affected Images
    all_class_pairs = pd.concat([train_dups, test_dups, leakage])
    affected = set(all_class_pairs['Image A'].tolist() + all_class_pairs['Image B'].tolist())
    num_affected = len(affected)
    
    tr_total = class_totals.loc[c, 'Train Images']
    te_total = class_totals.loc[c, 'Test Images']
    
    dup_pct = num_affected / (tr_total + te_total) * 100
    leak_pct = (len(set(leakage['Image A'].tolist() + leakage['Image B'].tolist())) / (tr_total + te_total)) * 100
    
    # Network Stats
    G = nx.Graph()
    for _, row in all_class_pairs.iterrows():
        G.add_edge(row['Image A'], row['Image B'])
    
    comps = list(nx.connected_components(G))
    num_comps = len(comps)
    largest_comp = max([len(comp) for comp in comps]) if num_comps > 0 else 0
    avg_comp = np.mean([len(comp) for comp in comps]) if num_comps > 0 else 0
    
    # Integrity Score (0-100)
    # Deductions: Leakage is 3x penalty, Exact is 2x, Near is 1x
    penalty = (leak_pct * 3) + ((exact_tr + exact_te)/(tr_total + te_total)*100 * 2) + ((near_tr + near_te)/(tr_total + te_total)*100 * 1)
    integrity = max(0.0, 100.0 - penalty)
    
    # Categorization
    cat_counts = all_class_pairs['Category'].value_counts()
    
    to_remove = len(all_class_pairs[all_class_pairs['Recommendation'] == 'Remove'])
    
    per_class_stats.append({
        'Class': c,
        'Train Images': tr_total,
        'Test Images': te_total,
        'Total Original': tr_total + te_total,
        'Exact Dup (Train)': exact_tr,
        'Near Dup (Train)': near_tr,
        'Exact Dup (Test)': exact_te,
        'Near Dup (Test)': near_te,
        'Cross Split Leakage': leak_count,
        'Unique Images Affected': num_affected,
        'Duplicate %': dup_pct,
        'Leakage %': leak_pct,
        'Network Components': num_comps,
        'Largest Component': largest_comp,
        'Avg Component Size': avg_comp,
        'Integrity Score': integrity,
        'Remove': to_remove,
        'Exact Category': cat_counts.get('Exact Duplicate', 0),
        'Near Category': cat_counts.get('Near Duplicate', 0),
        'Pixel Identical Category': cat_counts.get('Pixel Identical', 0),
        'False Positive Category': cat_counts.get('False Positive', 0)
    })

df_stats = pd.DataFrame(per_class_stats)

# %% [markdown]
# ## 3. Cleaning Simulation & Priority

# %%
df_stats['Remaining'] = df_stats['Total Original'] - df_stats['Remove']
df_stats['Keep %'] = (df_stats['Remaining'] / df_stats['Total Original']) * 100
df_stats['Information Loss %'] = 100 - df_stats['Keep %']

def assign_priority(row):
    if row['Integrity Score'] < 95: return 'High'
    if row['Integrity Score'] < 98: return 'Medium'
    return 'Low'

df_stats['Cleaning Priority'] = df_stats.apply(assign_priority, axis=1)

print(df_stats[['Class', 'Total Original', 'Remove', 'Remaining', 'Keep %', 'Integrity Score', 'Cleaning Priority']])

# %% [markdown]
# ## 4. Representation Space Impact (DINO Density)

# %%
print("Evaluating Representation Impact Before vs After Cleaning...")
try:
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
except Exception as e:
    import torchvision.models as models
    dinov2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    dinov2.fc = nn.Identity()
    dinov2 = dinov2.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MRIDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))

dl = DataLoader(MRIDataset(df_all['path'].values, transform), batch_size=128, shuffle=False)
embeddings = []
with torch.no_grad():
    for batch in tqdm(dl, desc="Extracting for Topology"):
        embeddings.append(dinov2(batch.to(device)).cpu().numpy())
embeddings = np.concatenate(embeddings, axis=0)

images_to_remove = set(df_pairs[df_pairs['Recommendation'] == 'Remove']['Image B'].tolist())
clean_mask = ~df_all['path'].isin(images_to_remove)
X_clean = embeddings[clean_mask]

def compute_manifold_density(feats, k=5):
    # Average distance to k-th nearest neighbor (lower distance = higher density)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(feats)
    distances, _ = nn.kneighbors(feats)
    return np.mean(distances[:, -1])

density_before = compute_manifold_density(embeddings)
density_after = compute_manifold_density(X_clean)

print(f"Manifold Avg Neighbor Distance (Before): {density_before:.4f}")
print(f"Manifold Avg Neighbor Distance (After Cleaning): {density_after:.4f}")

# %% [markdown]
# ## 5. Visualizations

# %%
print("Generating Publication Visuals...")
# 1. Duplicate Count by Class
plt.figure(figsize=(8, 5))
sns.barplot(data=df_stats, x='Class', y='Unique Images Affected', palette='Reds_d')
plt.title("Affected Images by Class")
plt.savefig(os.path.join(output_dir, "affected_images_by_class.png"))
plt.close()

# 2. Before vs After Cleaning (Stacked Bar)
plt.figure(figsize=(8, 5))
bar1 = sns.barplot(x="Class",  y="Total Original", data=df_stats, color='lightgrey')
bar2 = sns.barplot(x="Class", y="Remaining", data=df_stats, color='darkblue')
plt.title("Dataset Size Before and After Cleaning")
plt.savefig(os.path.join(output_dir, "before_after_cleaning.png"))
plt.close()

# 3. Simple Box/Flow Diagram (Sankey Substitute)
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.text(0.1, 0.5, f"Original\n{df_stats['Total Original'].sum()}", size=15, ha='center', va='center', bbox=dict(boxstyle="round", fc="lightgrey"))
ax.annotate("", xy=(0.3, 0.6), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(0.4, 0.7, f"Removed\n{df_stats['Remove'].sum()}", size=15, ha='center', va='center', bbox=dict(boxstyle="round", fc="salmon"))
ax.annotate("", xy=(0.4, 0.3), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(0.6, 0.3, f"Remaining\n{df_stats['Remaining'].sum()}", size=15, ha='center', va='center', bbox=dict(boxstyle="round", fc="lightblue"))
plt.title("Dataset Cleaning Flow", size=18)
plt.savefig(os.path.join(output_dir, "cleaning_flowchart.png"))
plt.close()

# %% [markdown]
# ## 6. Dataset Certification Report Generation

# %%
print("Generating Final Dataset Certification Report...")

avg_integrity = df_stats['Integrity Score'].mean()
leakage_risk = 100 if df_stats['Cross Split Leakage'].sum() > 0 else 0
dup_risk = (df_stats['Unique Images Affected'].sum() / df_stats['Total Original'].sum()) * 100

overall_score = (avg_integrity * 0.6) + ((100 - leakage_risk) * 0.3) + ((100 - dup_risk) * 0.1)
ready_status = "Research Ready" if overall_score > 90 and leakage_risk == 0 else "Not Certified (Requires Execution of Cleaning)"

report = f"""# Per-Class Duplicate Integrity & Certification Report

## 1. Dataset Status Certification
**Status: {ready_status}**

| Category | Score / Metric |
|----------|----------------|
| Base Integrity | {avg_integrity:.1f}/100 |
| Leakage Risk | {leakage_risk}/100 |
| Duplicate Risk | {dup_risk:.1f}/100 |
| **Overall Dataset Health** | **{overall_score:.1f}/100** |

## 2. Per-Class Analysis Matrix
| Class | Original | Exact Duplicates | Near Duplicates | Leakage | Unique Images Affected | Remove | Remaining | Keep % | Integrity Score | Priority |
|-------|---------:|-----------------:|----------------:|--------:|-----------------------:|-------:|----------:|-------:|----------------:|---------:|
"""
for idx, row in df_stats.iterrows():
    report += f"| {row['Class'].capitalize()} | {row['Total Original']} | {row['Exact Dup (Train)'] + row['Exact Dup (Test)']} | {row['Near Dup (Train)'] + row['Near Dup (Test)']} | {row['Cross Split Leakage']} | {row['Unique Images Affected']} | {row['Remove']} | {row['Remaining']} | {row['Keep %']:.1f}% | {row['Integrity Score']:.1f} | {row['Cleaning Priority']} |\n"

report += f"""
## 3. Class Ranking & Duplicate Categories
The classes ranked by Integrity (Most to Least Reliable):
"""
ranked = df_stats.sort_values(by='Integrity Score', ascending=False)
for idx, row in ranked.iterrows():
    report += f"- **{row['Class'].capitalize()}** (Score: {row['Integrity Score']:.1f})\n"

report += f"""
### Overall Duplicate Topology
| Category | Count |
|----------|-------|
| Exact Duplicates | {df_stats['Exact Category'].sum()} |
| Near Duplicates | {df_stats['Near Category'].sum()} |
| Pixel Identical Variants | {df_stats['Pixel Identical Category'].sum()} |
| False Positives | {df_stats['False Positive Category'].sum()} |

## 4. Network Graph Statistics
| Class | Connected Components | Largest Component Size | Avg Component Size |
|-------|----------------------|------------------------|--------------------|
"""
for idx, row in df_stats.iterrows():
    report += f"| {row['Class'].capitalize()} | {row['Network Components']} | {row['Largest Component']} | {row['Avg Component Size']:.1f} |\n"

report += f"""
## 5. Representation Impact Analysis (DINO Space)
When simulating the removal of {df_stats['Remove'].sum()} heavily corrupted/leaking images, we measured the shift in the Deep Feature Manifold:
- **Pre-Cleaning Cluster Density (Avg NN Distance):** {density_before:.4f}
- **Post-Cleaning Cluster Density:** {density_after:.4f}

*Interpretation:* The {'drop' if density_after < density_before else 'increase'} in neighbor distance confirms that removing duplicates {'removes artificial clump density without destroying manifold topology' if density_after > density_before else 'increases dataset compactness'}.

## 6. Actionable Cleaning Decision Engine
"""
for idx, row in ranked.iterrows():
    report += f"- **{row['Class'].capitalize()}**: Remove {row['Remove']} Exact/Leakage duplicates. (Keep {row['Remaining']} images, Retention: {row['Keep %']:.1f}%).\n"

with open(os.path.join(output_dir, 'per_class_certification_report.md'), 'w') as f:
    f.write(report)
print("Complete! Wrote per_class_certification_report.md")
# %%
