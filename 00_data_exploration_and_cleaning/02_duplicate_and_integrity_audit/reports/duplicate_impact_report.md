# Duplicate Impact Report & Verification Audit

## 1. Candidate Verification Summary Audit

| Stage                       | Candidate Pairs Remaining |
|-----------------------------|---------------------------|
| Initial pHash Candidates    | 1930                      |
| Deep Verification Queue     | 1930                      |
| Confirmed Exact Hashes      | 227                       |
| Pixel Identical             | 385                       |
| Near Duplicates (High SSIM) | 1008                      |
| False Positives             | 649                       |

## 2. Duplicate Categorization

| Category                       | Count |
|--------------------------------|-------|
| Near Duplicate                 | 732   |
| False Positive                 | 471   |
| Same Slice, Different Process  | 239   |
| Exact Duplicate                | 227   |
| Byte Different, Pixel Identical| 158   |
| Compression Difference         | 103   |

## 3. Leakage & Recommendations

- **Critical Leakage Risk Pairs (Score > 50):** 143
- **Images Recommended for Removal:** 318

## 4. Duplicate Impact Analysis (Ablation)

To determine if these duplicates scientifically impact representation learning, we performed an ablation study by training a Logistic Regression linear probe on frozen ResNet50 embeddings:

- **Baseline F1 Score (With Duplicates):** 0.9275
- **Cleaned F1 Score (Without Duplicates):** 0.9237
- **Performance Delta:** -0.0038

## 5. Research Interpretation

The multi-hash candidate funnel successfully narrowed down 7200 images to a subset of suspicious pairs. Geometric, structural (SSIM), and foundational (DINOv2/ResNet50) verifications revealed that `227` pairs are byte-level exact duplicates, while `103` pairs differ only via compression artifacts.

Crucially, the ablation study demonstrates a performance delta of `-0.0038`. If this delta is negligible, the duplicates do not artificially inflate the feature-space density in a way that biases a linear classifier. However, the `143` pairs exhibiting severe train-test leakage pose a critical risk to generalization validity and **MUST** be removed before clinical evaluation.
