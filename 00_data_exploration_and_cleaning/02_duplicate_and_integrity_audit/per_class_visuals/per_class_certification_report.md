# Per-Class Duplicate Integrity & Certification Report

## 1. Dataset Status Certification

**Status: Not Certified (Requires Execution of Cleaning)**

| Category             | Score / Metric |
|----------------------|----------------|
| Base Integrity       | 76.3/100       |
| Leakage Risk         | 100/100        |
| Duplicate Risk       | 17.3/100       |
| **Overall Dataset Health** | **54.1/100**|

## 2. Per-Class Analysis Matrix

| Class             | Original | Exact Duplicates | Near Duplicates | Leakage | Unique Images Affected | Remove | Remaining | Keep % | Integrity Score | Priority |
|-------------------|---------:|-----------------:|----------------:|--------:|-----------------------:|-------:|----------:|-------:|----------------:|---------:|
| Glioma            | 1800     | 18               | 0               | 0       | 40                     | 18     | 1782      | 99.0%  | 98.0            | Low      |
| Meningioma        | 1800     | 19               | 0               | 6       | 72                     | 25     | 1775      | 98.6%  | 96.2            | Medium.  |
| Notumor           | 1800     | 169              | 420             | 142     | 1049                   | 311    | 1489      | 82.7%  | 17.6            | High     |
| Pituitary         | 1800     | 58               | 0               | 0       | 82                     | 58     | 1742      | 96.8%  | 93.6            | High     |

## 3. Class Ranking & Duplicate Categories

The classes ranked by Integrity (Most to Least Reliable):

- **Glioma** (Score: 98.0)
- **Meningioma** (Score: 96.2)
- **Pituitary** (Score: 93.6)
- **Notumor** (Score: 17.6)

### Overall Duplicate Topology

| Category                 | Count |
|--------------------------|-------|
| Exact Duplicates         | 227   |
| Near Duplicates          | 420   |
| Pixel Identical Variants | 37    |
| False Positives          | 546   |

## 4. Network Graph Statistics

| Class             | Connected Components | Largest Component Size | Avg Component Size |
|-------------------|----------------------|------------------------|--------------------|
| Glioma            | 19                   | 3                      | 2.1                |
| Meningioma        | 33                   | 4                      | 2.2                |
| Notumor           | 369                  | 18                     | 2.8                |
| Pituitary         | 32                   | 6                      | 2.6                |

## 5. Representation Impact Analysis (DINO Space)

When simulating the removal of 412 heavily corrupted/leaking images, we measured the shift in the Deep Feature Manifold:
- **Pre-Cleaning Cluster Density (Avg NN Distance):** = 17.9738
- **Post-Cleaning Cluster Density:** 18.3219

*Interpretation:* The increase in neighbor distance confirms that removing duplicates removes artificial clump density without destroying manifold topology.

## 6. Actionable Cleaning Decision Engine

- **Glioma**: Remove 18 Exact/Leakage duplicates. (Keep 1782 images, Retention: 99.0%).
- **Meningioma**: Remove 25 Exact/Leakage duplicates. (Keep 1775 images, Retention: 98.6%).
- **Pituitary**: Remove 58 Exact/Leakage duplicates. (Keep 1742 images, Retention: 96.8%).
- **Notumor**: Remove 311 Exact/Leakage duplicates. (Keep 1489 images, Retention: 82.7%).
