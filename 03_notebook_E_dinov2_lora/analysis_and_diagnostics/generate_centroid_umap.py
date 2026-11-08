import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

def main():
    EXP_DIR = Path("outputs/EXP_LORA")
    data = np.load(EXP_DIR / "features/lora_test_features.npz")
    features = data['features']
    labels = data['labels']
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    print("Computing 2D UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    
    # 1. Plot the scatter points (dot dot dot)
    colors = sns.color_palette('deep', n_colors=4)
    for i, name in enumerate(class_names):
        idx = labels == i
        plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1], label=name, color=colors[i], alpha=0.3, s=20)
        
    # 2. Calculate and plot centroids in 2D space
    centroids_2d = {}
    for i, name in enumerate(class_names):
        idx = labels == i
        centroid = umap_emb[idx].mean(axis=0)
        centroids_2d[name] = centroid
        
        # Plot large centroid marker
        plt.scatter(centroid[0], centroid[1], color=colors[i], marker='X', s=400, edgecolors='black', linewidth=2, zorder=5)
        plt.text(centroid[0], centroid[1] + 0.3, name.upper(), fontsize=12, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 3. Draw lines between centroids
    keys = list(centroids_2d.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            c1 = centroids_2d[keys[i]]
            c2 = centroids_2d[keys[j]]
            
            # Draw line
            plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'k--', alpha=0.5, zorder=3)
            
            # We will just draw the lines to visually show the graph structure. 
            # We don't annotate distances because 2D UMAP distances don't perfectly match the 768-D Euclidean distances.

    plt.title("Inter-Class Cluster Separations (LoRA DINOv2 Features)", fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Custom legend for centroids
    import matplotlib.lines as mlines
    centroid_marker = mlines.Line2D([], [], color='black', marker='X', linestyle='None',
                              markersize=10, label='Class Centroid')
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    handles.append(centroid_marker)
    plt.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.savefig("/Users/shashankchaurasia/.gemini/antigravity-ide/brain/6e2eb918-6ec5-4ba5-8b8e-fbad4ceb8cae/interclass_cluster_visualization.png", dpi=300)
    print("Saved to artifacts!")

if __name__ == "__main__":
    main()
