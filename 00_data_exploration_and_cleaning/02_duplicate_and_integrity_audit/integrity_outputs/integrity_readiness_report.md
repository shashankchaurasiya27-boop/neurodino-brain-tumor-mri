# Dataset Integrity, Redundancy, and Leakage Analysis

## 1. Publication Readiness Scorecard

| Category | Score / Metric |
|----------|----------------|

| Effective Dataset Quality | 94.0/100 |
| Total Images Analyzed | 7200 |
| Images Flagged for Removal | 433 |
| Train-Test Leakage Pairs | 148 |
| Inferred Patient Clusters | 388 |
| Metadata Bias Risk (Acc) | 78.8% |
| **MICCAI/TMI Readiness** | **No (Requires Cleaning & Debiasing)** |

## 2. Statistical Duplicate Ablation Study

We performed a statistically rigorous ablation study measuring the linear separability of deep foundation features (DINOv2) before and after removing 433 corrupted artifacts.

- **Baseline F1 Score:** 0.9625 [95% CI: 0.9522, 0.9727]
- **Cleaned Dataset F1:** 0.9605 [95% CI: 0.9491, 0.9718]
- **Mann-Whitney U p-value:** 7.1154e-03
- **Effect Size (Cohen's d):** -0.3662

*Conclusion on Ablation:* The dataset representation changed significantly.

## 3. Automatic Reviewer Simulation

**Q: Does Train-Test leakage exist, and does it invalidate deep learning generalization on this dataset?**
A: Yes. We detected 148 cross-split leakage instances. This absolutely invalidates any test accuracies derived from the raw Kaggle format.

**Q: Are the 'duplicates' merely adjacent MRI slices from the same patient?**
A: By enforcing Brain Mask Overlap (IoU) and DINOv2 $>99.999\%$ embedding similarity, the Decision Engine algorithm specifically distinguished artifactual duplications from volumetric adjacent slices.

## 4. Research Opportunity Generator

- **Observation**: The dataset exhibits a Metadata Bias Risk of 78.8%.
- **Evidence**: A Random Forest classifier predicted the tumor class solely based on non-anatomical metadata (Brightness, Resolution, Entropy) well above the chance rate.
- **Research Opportunity**: Bias-invariant Medical Representation Learning.
- **Future Experiment**: Train a Domain-Adversarial Neural Network (DANN) to unlearn brightness/resolution correlations while maintaining tumor separability.

## 5. Dataset Cleaning Verdict

The raw dataset contains critical data leakage and systemic representation redundancy. The generated `duplicate_decision_matrix.csv` must be used to purge the 433 flagged images before any novel architecture is trained.
