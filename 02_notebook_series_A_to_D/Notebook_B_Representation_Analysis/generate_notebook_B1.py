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

# --- INTRODUCTION ---
cells.append(create_markdown_cell("""# Notebook B1: Representation Space Analysis
**Scientific Protocol**
This notebook acts as the Representation Laboratory. 
*Purpose*: Measure, Analyze, Visualize, Certify the frozen DINOv2 latent space.
*Constraint*: Tumor labels are strictly isolated during geometric structure measurement to prevent label leakage.
"""))

cells.append(create_code_cell("""import os
import sys
import json
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist, pdist, squareform, mahalanobis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

try:
    import umap
except ImportError:
    os.system('pip install umap-learn')
    import umap

# Discover Latest Experiment Output
output_dirs = sorted(glob.glob("outputs/EXP_*"), reverse=True)
if not output_dirs:
    raise FileNotFoundError("No Notebook A outputs found!")
EXP_DIR = Path(output_dirs[0])
print(f"Loading features from: {EXP_DIR}")

# Load Files
train_feat = np.load(EXP_DIR / 'features/train_features.npy')
test_feat = np.load(EXP_DIR / 'features/test_features.npy')
npz_data = np.load(EXP_DIR / 'features/dataset_features.npz', allow_pickle=True)
train_lbl = npz_data['train_lbl']
test_lbl = npz_data['test_lbl']
metadata_df = pd.read_csv(EXP_DIR / 'configs/dataset_manifest.csv')

features = np.vstack([train_feat, test_feat])
labels = np.concatenate([train_lbl, test_lbl])
classes = np.unique(labels)

b1_metrics = {}
"""))

# --- PHASE 1: FEATURE CERTIFICATION ---
cells.append(create_markdown_cell("""## Phase 1: Feature Certification
**Research Question**: Are the generated features mathematically sound and complete?
**Methodology**: Compute L2 norms, detect missing values/zero vectors, and verify dimensionality.
"""))

cells.append(create_code_cell("""l2_norms = np.linalg.norm(features, axis=1)
zero_vectors = np.sum(l2_norms == 0)
nans = np.isnan(features).sum()
infs = np.isinf(features).sum()

print("--- Feature Certification ---")
print(f"Feature Dimension: {features.shape[1]}")
print(f"NaN Values: {nans}")
print(f"Inf Values: {infs}")
print(f"Zero Vectors: {zero_vectors}")
print(f"Mean L2 Norm: {l2_norms.mean():.4f} +/- {l2_norms.std():.4f}")

b1_metrics["feature_certification"] = {
    "dim": features.shape[1],
    "zero_vectors": int(zero_vectors),
    "nans": int(nans),
    "mean_l2": float(l2_norms.mean())
}
"""))

# --- PHASE 2: INTRINSIC DIMENSIONALITY ---
cells.append(create_markdown_cell("""## Phase 2: Intrinsic Dimensionality Analysis
**Research Question**: How complex is the representation space? 
**Methodology**: PCA explained variance, Participation Ratio, and Scree Plot.
"""))

cells.append(create_code_cell("""pca = PCA().fit(features)
expl_var = pca.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

# Participation Ratio (PR)
eigenvalues = pca.explained_variance_
participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)

d90 = np.argmax(cum_var >= 0.90) + 1
d95 = np.argmax(cum_var >= 0.95) + 1
d99 = np.argmax(cum_var >= 0.99) + 1

print("--- Intrinsic Dimensionality ---")
print(f"Dimensions for 90% variance: {d90}")
print(f"Dimensions for 95% variance: {d95}")
print(f"Participation Ratio (Continuous ID): {participation_ratio:.2f}")

b1_metrics["intrinsic_dimensionality"] = {
    "d90": int(d90),
    "participation_ratio": float(participation_ratio)
}
"""))

# --- PHASE 3: FEATURE REDUNDANCY ---
cells.append(create_markdown_cell("""## Phase 3: Feature Redundancy Analysis
**Research Question**: How redundant are the latent features?
**Methodology**: Singular Value Spectrum, Condition Number, Effective Rank, Feature Sparsity.
"""))

cells.append(create_code_cell("""u, s, vh = np.linalg.svd(features, full_matrices=False)
condition_number = s[0] / s[-1]
effective_rank = np.sum(s > 1e-5 * s[0])
sparsity = np.mean(np.abs(features) < 1e-4)

print("--- Feature Redundancy ---")
print(f"Condition Number: {condition_number:.2f}")
print(f"Effective Rank: {effective_rank} / {features.shape[1]}")
print(f"Feature Sparsity: {sparsity*100:.2f}%")

b1_metrics["feature_redundancy"] = {
    "condition_number": float(condition_number),
    "effective_rank": int(effective_rank),
    "sparsity": float(sparsity)
}
"""))

# --- PHASE 4: REPRESENTATION DRIFT ---
cells.append(create_markdown_cell("""## Phase 4: Representation Drift Analysis
**Research Question**: Are local and global topologies preserved across neighborhoods?
**Methodology**: Measure neighbor distances across PCA projections. Compare original features against nearest 100 neighbors.
"""))

cells.append(create_code_cell("""nn = NearestNeighbors(n_neighbors=100).fit(features)
distances, indices = nn.kneighbors(features)

mean_local_radius = np.mean(distances[:, 1:10]) # Local geometry
mean_global_radius = np.mean(distances[:, -10:]) # Global topology drift

print(f"Mean 10-NN Radius (Local Geometry Spread): {mean_local_radius:.4f}")
print(f"Mean 100-NN Radius (Global Topology Spread): {mean_global_radius:.4f}")

b1_metrics["representation_drift"] = {
    "local_geometry": float(mean_local_radius),
    "global_topology": float(mean_global_radius)
}
"""))

# --- PHASE 5: UNSUPERVISED VISUALIZATION ---
cells.append(create_markdown_cell("""## Phase 5: Unsupervised Visualization
**Constraint**: NO TUMOR LABELS are provided to these algorithms.
"""))

cells.append(create_code_cell("""print("Computing PCA, t-SNE, UMAP (Unsupervised)...")
pca_2d = PCA(n_components=2).fit_transform(features)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(features)
"""))

# --- PHASE 6: SUPERVISED VISUALIZATION OVERLAY ---
cells.append(create_markdown_cell("""## Phase 6: Supervised Visualization Overlay
**Methodology**: Now overlay the ground truth labels onto the unsupervised geometries.
"""))

cells.append(create_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
methods = [('PCA', pca_2d), ('t-SNE', tsne_2d), ('UMAP', umap_2d)]

for ax, (name, emb) in zip(axes, methods):
    df_plot = pd.DataFrame({'x': emb[:,0], 'y': emb[:,1], 'Label': labels})
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Label', s=5, ax=ax, palette='Set1', alpha=0.7)
    ax.set_title(f"{name} Overlay")
    ax.axis('off')

plt.tight_layout()
plt.savefig(EXP_DIR / 'figures/unsupervised_geometries.png', dpi=300)
"""))

# --- PHASE 7: LINEAR SEPARABILITY SCORE ---
cells.append(create_markdown_cell("""## Phase 7: Linear Separability Score
**Research Question**: Are the representations fundamentally linearly separable?
**Methodology**: Train LinearSVC. Measure Decision Boundary, Class Margins.
"""))

cells.append(create_code_cell("""svc = LinearSVC(dual=False, random_state=42)
svc.fit(train_feat, train_lbl)
acc = svc.score(test_feat, test_lbl)

# Decision Boundary and Margin analysis
margins = svc.decision_function(test_feat)
# Distance to decision boundary (Margin)
mean_margin = np.mean(np.abs(margins))
# Class Margin (Confidence delta between top 1 and top 2 predictions)
class_margin = np.mean(np.max(margins, axis=1) - np.sort(margins, axis=1)[:, -2])

print(f"Linear Probe Accuracy: {acc*100:.2f}%")
print(f"Mean Decision Boundary Margin: {mean_margin:.4f}")
print(f"Class Confidence Margin: {class_margin:.4f}")

b1_metrics["linear_separability"] = {
    "probe_accuracy": float(acc),
    "decision_boundary_margin": float(mean_margin),
    "class_margin": float(class_margin)
}
"""))

# --- PHASE 8: CLASS SEPARABILITY ---
cells.append(create_markdown_cell("""## Phase 8: Class Separability
**Methodology**: Compute Mahalanobis Distance, Bhattacharyya Distance, Fisher Discriminant Ratio, and Cluster Density.
"""))

cells.append(create_code_cell("""centroids = {c: features[labels == c].mean(axis=0) for c in classes}
covariances = {c: np.cov(features[labels == c], rowvar=False) + np.eye(features.shape[1])*1e-4 for c in classes}
inv_cov = {c: np.linalg.inv(covariances[c]) for c in classes}

# Mahalanobis and Bhattacharyya
maha_matrix = np.zeros((len(classes), len(classes)))
bhat_matrix = np.zeros((len(classes), len(classes)))

for i, c1 in enumerate(classes):
    for j, c2 in enumerate(classes):
        if i == j: continue
        diff = centroids[c1] - centroids[c2]
        # Mahalanobis
        maha_matrix[i, j] = np.sqrt(diff.T @ inv_cov[c1] @ diff)
        # Bhattacharyya
        cov_mean = (covariances[c1] + covariances[c2]) / 2.0
        try:
            sign_det, log_det_mean = np.linalg.slogdet(cov_mean)
            _, log_det_1 = np.linalg.slogdet(covariances[c1])
            _, log_det_2 = np.linalg.slogdet(covariances[c2])
            bhat_dist = 0.125 * diff.T @ np.linalg.inv(cov_mean) @ diff + 0.5 * (log_det_mean - 0.5*(log_det_1 + log_det_2))
            bhat_matrix[i, j] = bhat_dist
        except:
            bhat_matrix[i, j] = np.nan

# Fisher Discriminant Ratio & Cluster Density
global_mean = features.mean(axis=0)
inter_var = np.sum([np.sum(labels==c) * np.linalg.norm(centroids[c]-global_mean)**2 for c in classes])
intra_var = np.sum([np.trace(covariances[c]) * np.sum(labels==c) for c in classes])
fdr = inter_var / intra_var

# Cluster Density (Inverse of average intra-class variance)
cluster_density = 1.0 / (intra_var / len(features))

print(f"Fisher Discriminant Ratio: {fdr:.4f}")
print(f"Mean Cluster Density: {cluster_density:.4f}")
print(f"Mean Bhattacharyya Distance: {np.nanmean(bhat_matrix):.4f}")

b1_metrics["class_separability"] = {
    "fdr": float(fdr),
    "cluster_density": float(cluster_density),
    "mean_bhattacharyya": float(np.nanmean(bhat_matrix))
}
"""))

# --- PHASE 9: PROTOTYPE ANALYSIS ---
cells.append(create_markdown_cell("""## Phase 9: Prototype Analysis
**Methodology**: Compute Prototype Stability, Bootstrap Prototypes, Prototype Drift, Prototype Radius, and Confidence.
"""))

cells.append(create_code_cell("""# Bootstrap Prototypes
n_bootstraps = 50
prototype_stabilities = {}
prototype_drifts = {}
prototype_radii = {}

for c in classes:
    feats_c = features[labels == c]
    # Radius (95% containment)
    dists = cdist(feats_c, [centroids[c]], metric='euclidean')
    prototype_radii[c] = float(np.percentile(dists, 95))
    
    # Bootstrap
    boot_centroids = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(feats_c), size=len(feats_c), replace=True)
        boot_centroids.append(np.mean(feats_c[idx], axis=0))
    boot_centroids = np.array(boot_centroids)
    
    # Stability (Variance of bootstrapped centroids)
    prototype_stabilities[c] = float(1.0 / (np.mean(np.var(boot_centroids, axis=0)) + 1e-6))
    
    # Drift (Max distance between bootstraps)
    drift_dists = pdist(boot_centroids, metric='euclidean')
    prototype_drifts[c] = float(np.max(drift_dists))

mean_stability = np.mean(list(prototype_stabilities.values()))
print(f"Mean Prototype Stability: {mean_stability:.2f}")

b1_metrics["prototype_analysis"] = {
    "radii": prototype_radii,
    "stabilities": prototype_stabilities,
    "drifts": prototype_drifts,
    "mean_stability": float(mean_stability)
}
"""))

# --- PHASE 10: NEIGHBORHOOD ANALYSIS ---
cells.append(create_markdown_cell("""## Phase 10: Neighborhood Analysis
**Methodology**: Compute Trustworthiness and Continuity.
"""))

cells.append(create_code_cell("""# Trustworthiness
trust = trustworthiness(features[:2000], tsne_2d[:2000], n_neighbors=5)
# Continuity (trustworthiness of the inverse map)
cont = trustworthiness(tsne_2d[:2000], features[:2000], n_neighbors=5)

print(f"t-SNE Trustworthiness: {trust:.4f}")
print(f"t-SNE Continuity: {cont:.4f}")

b1_metrics["neighborhood_analysis"] = {
    "trustworthiness": float(trust),
    "continuity": float(cont)
}
"""))

# --- PHASE 11: FEATURE SPACE CERTIFICATION ---
cells.append(create_markdown_cell("""## Phase 11: Feature Space Certification
**Expected Output**: JSON payload `b1_metrics.json`.
"""))

cells.append(create_code_cell("""with open(EXP_DIR / 'reports/b1_metrics.json', 'w') as f:
    json.dump(b1_metrics, f, indent=4)
print("Notebook B1 Execution Complete. Metrics rigorously saved for Notebook B2.")
"""))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 5
}
with open('Notebook_B1_Representation_Space_Analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
print("Successfully generated Notebook_B1_Representation_Space_Analysis.ipynb")
