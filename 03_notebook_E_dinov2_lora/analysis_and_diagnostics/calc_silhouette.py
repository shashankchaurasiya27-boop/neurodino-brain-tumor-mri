import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_score

def main():
    EXP_DIR = Path("outputs/EXP_LORA")
    data = np.load(EXP_DIR / "features/lora_test_features.npz")
    features = data['features']
    labels = data['labels']
    
    score = silhouette_score(features, labels)
    print(f"Silhouette Score: {score:.4f}")

if __name__ == "__main__":
    main()
