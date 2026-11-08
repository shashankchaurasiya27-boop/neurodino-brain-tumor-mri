# Brain Tumor MRI - Top-Tier Research Report & Protocol

## Phase 1-5: Data Integrity & Anatomy

- Total Valid Images: 7200
- Corrupted Images: 0
- Near Duplicates Found: 853
- Average Brain Symmetry (IoU): 0.751

## Phase 6-10: Feature Space & Bias

- Extracted Embeddings Shape: (7200, 2048)
- UMAP Trustworthiness: 0.979
- Metadata Predictivity (Bias Risk): 100.0%
- Isolation Forest Outliers: 72
- Local Outlier Factor (LOF) Outliers: 72

## Phase 11: Linear Probe Separability

| Model               | F1 Score | Cohen's Kappa |
|---------------------|----------|---------------|
| Logistic Regression | 0.930    | 0.906         |
| Linear SVM          | 0.913    | 0.883         |
| Ridge Classifier    | 0.913    | 0.884         |
| kNN (k=5)           | 0.860    | 0.819         |
| Nearest Centroid    | 0.704    | 0.617         |

## Phase 13: Auto-Generated Research Hypotheses

### Hypothesis 1: Severe Dataset Bias Exists

- **Evidence**: Metadata (resolution, brightness) alone predicts class with 100.0% accuracy.
- **Suggested Experiment**: Debias the dataset by equalizing resolution and intensity histograms before training any CNN.
- **Suggested Architecture**: Use Domain Adversarial Neural Networks (DANN) to unlearn metadata bias.

### Hypothesis 2: Deep CNNs may be overkill

- **Evidence**: A linear probe on ImageNet embeddings achieves F1=0.930. The classes are linearly separable.
- **Suggested Experiment**: Compare a simple Logistic Regression on embeddings vs a fully fine-tuned ViT. If margins are <3%, prioritize lightweight models.
- **Suggested Architecture**: Lightweight MobileNetV3 or purely traditional ML on deep features.

### Hypothesis 3: Natural clustering correlates strongly with MRI Orientation

- **Evidence**: UMAP trustworthiness is very high (0.979), implying strong local neighborhood preservation.
- **Suggested Experiment**: Perform unsupervised orientation discovery (Axial/Coronal/Sagittal) and train separate expert models for each view.
- **Suggested Architecture**: Mixture of Experts (MoE) where gating network routes based on anatomical orientation.