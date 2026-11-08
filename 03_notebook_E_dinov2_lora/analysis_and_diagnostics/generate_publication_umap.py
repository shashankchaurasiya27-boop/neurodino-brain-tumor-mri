import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path

def main():
    EXP_DIR = Path("outputs/EXP_LORA")
    data = np.load(EXP_DIR / "features/lora_test_features.npz")
    features = data['features']
    labels = data['labels']
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    print("Computing 2D UMAP for Publication...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(features)
    
    # Set publication-quality style
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette('colorblind', n_colors=4)
    
    for i, name in enumerate(class_names):
        idx = labels == i
        # Use a slightly higher alpha and nice edge colors for a clean journal look
        plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1], label=name, 
                    color=colors[i], alpha=0.7, s=40, edgecolor='white', linewidth=0.5)
        
    plt.title("LoRA DINOv2 Feature Space (Test Set)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("UMAP Dimension 1", fontsize=14, fontweight='bold')
    plt.ylabel("UMAP Dimension 2", fontsize=14, fontweight='bold')
    
    # Clean, professional legend
    plt.legend(title="Tumor Class", title_fontsize='13', loc='best', 
               frameon=True, shadow=True, fancybox=True, fontsize=12)

    plt.tight_layout()
    
    output_path = "/Users/shashankchaurasia/.gemini/antigravity-ide/brain/6e2eb918-6ec5-4ba5-8b8e-fbad4ceb8cae/lora_umap_publication.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight') # 600 DPI for journal quality
    print(f"Saved publication graph to {output_path}")

if __name__ == "__main__":
    main()
