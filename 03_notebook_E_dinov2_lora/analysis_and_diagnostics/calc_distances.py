import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

def main():
    EXP_DIR = Path("outputs/EXP_LORA")
    data = np.load(EXP_DIR / "features/lora_test_features.npz")
    features = data['features']
    labels = data['labels']
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    print("--- Intra-Class Distances (Average distance of points to their class centroid) ---")
    centroids = {}
    for i, name in enumerate(class_names):
        class_feats = features[labels == i]
        centroid = class_feats.mean(axis=0)
        centroids[name] = centroid
        
        # Intra-class distance (mean Euclidean distance to centroid)
        dists = np.linalg.norm(class_feats - centroid, axis=1)
        print(f"{name:15s} : {dists.mean():.4f}")
        
    print("\n--- Inter-Class Distances (Distance between class centroids) ---")
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            name1 = class_names[i]
            name2 = class_names[j]
            dist = np.linalg.norm(centroids[name1] - centroids[name2])
            print(f"{name1:15s} vs {name2:15s} : {dist:.4f}")

if __name__ == "__main__":
    main()
