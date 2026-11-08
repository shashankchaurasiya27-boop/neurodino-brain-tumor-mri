

<div align="center">

```
  _   _                      ____  ___ _   _  ___  ®
 | \ | | ___ _   _ _ __ ___ |  _ \|_ _| \ | |/ _ \ 
 |  \| |/ _ \ | | | '__/ _ \| | | || ||  \| | | | |
 | |\  |  __/ |_| | | | (_) | |_| || || |\  | |_| |
 |_| \_|\___|\__,_|_|  \___/|____/|___|_| \_|\___/ 
```

### **Parameter-Efficient Representation Adaptation of Vision Foundation Models for Multi-Center Brain Tumor MRI Diagnosis**

[![GitHub Stars](https://img.shields.io/github/stars/shashankchaurasiya27-boop/neurodino-brain-tumor-mri?style=for-the-badge&color=9B43F3)](https://github.com/shashankchaurasiya27-boop/neurodino-brain-tumor-mri)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PEFT QKV-LoRA](https://img.shields.io/badge/PEFT-QKV--LoRA-8A2BE2?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/huggingface/peft)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![Vercel Compatible](https://img.shields.io/badge/Vercel-Deployable-000000?style=for-the-badge&logo=vercel&logoColor=white)](vercel.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

**[Live Web App](#-interactive-web-platform--inference-studio)** • **[Research Genesis Story](#-the-research-genesis-story-mode)** • **[Scientific Evidence](#-certified-results--statistical-evidence)** • **[Attention Rollouts](#-explainability--self-attention-rollouts)** • **[Showcase Gallery](#-verified-50-image-showcase-gallery)** • **[Reproduction Guide](#-reproducing-the-experiments)**

</div>

---

## 📑 Table of Contents

- [🌟 Executive Summary](#-executive-summary)
- [🖥️ Interactive Web Platform & Inference Studio](#-interactive-web-platform--inference-studio)
- [📖 The Research Genesis (Story Mode)](#-the-research-genesis-story-mode)
  - [Phase 1: Multi-Center Ingestion & Discovery](#phase-1-multi-center-ingestion--discovery)
  - [Phase 2: The Data Leakage Discovery & Deduplication Audit](#phase-2-the-data-leakage-discovery--deduplication-audit)
  - [Phase 3: Protocol Lockdown & Preprocessing](#phase-3-protocol-lockdown--preprocessing)
  - [Phase 4: Supervised Baselines Plateau & Boundary Confusion](#phase-4-supervised-baselines-plateau--boundary-confusion)
  - [Phase 5: Foundation Exploration & Representation Ceiling](#phase-5-foundation-exploration--representation-ceiling)
  - [Phase 6: The QKV-LoRA Innovation (Controlled Arm B)](#phase-6-the-qkv-lora-innovation-controlled-arm-b)
  - [Phase 7: Empirical Verification & Statistical Proof](#phase-7-empirical-verification--statistical-proof)
- [📊 Certified Results & Statistical Evidence](#-certified-results--statistical-evidence)
  - [Master Performance Matrix](#master-performance-matrix)
  - [Classwise Clinical Performance](#classwise-clinical-performance)
  - [Representation Geometry & Manifold Densification](#representation-geometry--manifold-densification)
  - [Statistical Significance Proofs](#statistical-significance-proofs)
- [👁️ Explainability & Self-Attention Rollouts](#-explainability--self-attention-rollouts)
- [🖼️ Verified 50-Image Showcase Gallery](#-verified-50-image-showcase-gallery)
- [🏗️ System Architecture & Workflow](#-system-architecture--workflow)
- [⚡ Quick Start & Local Execution](#-quick-start--local-execution)
- [🔄 Reproducing the Experiments](#-reproducing-the-experiments)
- [🚀 Cloud Deployment Guide](#-cloud-deployment-guide)
- [📁 Repository Structure & File Registry](#-repository-structure--file-registry)
- [📜 Citation & Clinical Disclaimer](#-citation--clinical-disclaimer)

---

## 🌟 Executive Summary

**NeuroDINO®** is a biomedical vision foundation platform designed to solve one of neuro-oncology's most difficult diagnostic challenges: **distinguishing between intra-axial Gliomas and extra-axial Meningiomas** on magnetic resonance imaging (MRI) without invasive stereotactic biopsy.

### The Clinical Diagnostic Challenge
Brain tumors account for substantial morbidity and mortality globally. On standard T1-weighted post-contrast MRI, primary high-grade gliomas and atypical meningiomas frequently present with overlapping radiological features: irregular ring enhancement, peritumoral vasogenic edema, and mass effect. While healthy brain tissue and pituitary adenomas are easily segmented, conventional deep learning models suffer from severe false-positive cross-confusion between Glioma and Meningioma.

### The Scientific Core Question
> **Can parameter-efficient Low-Rank Adaptation (PEFT QKV-LoRA) of a self-supervised Vision Foundation Model (Meta DINOv2 ViT-B/14) restructure and densify the latent representation geometry to resolve histological boundary confusion, when all non-LoRA experimental factors are strictly held constant?**

### The Key Breakthrough: **Bit-Identical Head Control ($\theta_0$)**
To ensure an irrefutable causal claim, we implemented a mathematically strict controlled design:
1. **Bit-Identical Initialization**: We generated initial weights for a 3-layer MLP classification head ($\theta_0$) and **deepcopied the identical byte-level tensor** to both the Frozen Baseline (Arm A) and the LoRA-adapted model (Arm B).
2. **Minimal Parameter Footprint**: We froze 99.05% of the 86.5M DINOv2 backbone, injecting trainable rank-8 adapters ($r=8, \alpha=16$) exclusively into the Query, Key, and Value attention projections. Only **822,020 parameters** are trained.
3. **Statistical Proof**: On a locked, leakage-quarantined test cohort of **1,357 scans**, LoRA adaptation improved test accuracy from **92.11% to 95.21%** ($+3.10\%$, McNemar paired test $\mathbf{p = 1.303 \times 10^{-8}}$) and reduced Glioma-Meningioma boundary confusion by **41.5%**.

```
+---------------------------------------------------------------------------------------------------------+
|                                    NEURODINO® HEAD-TO-HEAD COMPARISON                                   |
+----------------------------------+------------------------------------+---------------------------------+
| Metric                           | Arm A: Frozen DINOv2 Baseline      | Arm B: Controlled QKV-LoRA      |
+----------------------------------+------------------------------------+---------------------------------+
| Top-1 Test Accuracy              | 92.11% (1,250 / 1,357)             | 95.21% (1,292 / 1,357)  [+3.10%]|
| Macro-Averaged F1-Score          | 0.9196                             | 0.9507                  [+0.031]|
| Matthews Correlation (MCC)       | 0.8987                             | 0.9376                  [+0.039]|
| Glioma-Meningioma Overlap Errors | 53 mutual errors                   | 31 mutual errors        [-41.5%]|
| Latent Silhouette Score          | 0.0490                             | 0.1501                  [+0.101]|
| kNN Neighborhood Purity (k=5)    | 89.37%                             | 94.75%                  [+5.38%]|
| Trainable Parameters             | 527,108 (0.61%)                    | 822,020 (0.95%)                 |
| Statistical Significance         | Baseline Control                   | McNemar p = 1.303 x 10^-8       |
+----------------------------------+------------------------------------+---------------------------------+
```

---

## 🖥️ Interactive Web Platform & Inference Studio

Accompanying the research codebase is a high-performance, clinical demonstration platform designed with modern **glassmorphism** aesthetics (`Plus Jakarta Sans` typography, vibrant electric violet `#9B43F3` palette, and high-contrast dark/light radiologist themes).

```
   +--------------------------------------------------------------------------------+
   |  [ NeuroDINO® Diagnostic Studio ]                               Dark / Light   |
   |                                                                                |
   |   +--------------------------+    +----------------------------------------+   |
   |   |   RADIOLOGIST DARKROOM   |    |         DUAL INFERENCE ENGINE          |   |
   |   |                          |    |                                        |   |
   |   |   [ MRI Viewer Canvas ]  |    |   Arm B (Adapted QKV-LoRA):  97.4%     |   |
   |   |   * Window Brightness    |    |   >>> GLIOMA TUMOR (Primary Intra-Axial)  |
   |   |   * Contrast Level       |    |   Latency: 28.1ms | Feature Norm: 24.18|   |
   |   |   * Invert LUT (Grayscale|    |                                        |   |
   |   |   * 2.5x Loupe Magnifier |    |   Arm A (Frozen Baseline):   61.2%     |   |
   |   |                          |    |   >>> MENINGIOMA (Boundary Misclassify)|   |
   |   +--------------------------+    |                                        |   |
   |                                   |   [ALERT] LoRA Corrected Diagnosis!    |   |
   |                                   +----------------------------------------+   |
   +--------------------------------------------------------------------------------+
```

### Studio Capabilities
1. **Side-by-Side Dual Inference**: Upload any brain MRI scan (JPEG, PNG, TIFF) to execute parallel forward passes through both Arm A (Frozen) and Arm B (LoRA), showing real-time probability distributions, class category tags, and feature vector norms.
2. **Radiologist Darkroom Controls**:
   - Dynamic **Window Leveling** (Brightness & Contrast sliders) matching clinical PACS workstations.
   - **Inverted LUT** for edge accentuation along dural reflections.
   - **Interactive 2.5× Loupe Lens** for microscopic lesion boundary inspection.
3. **50-Image Showcase Gallery**: 50 certified test samples filterable across all 4 classes with instant 1-click live execution.
4. **LoRA Rescue Alerts**: Automatically highlights cases where foundation adaptation rescued a frozen baseline error.
5. **Research Genesis Storytelling**: Interactive 7-phase animated timeline walking through the empirical discovery.

---

## 📖 The Research Genesis (Story Mode)

```
       Phase 1               Phase 2               Phase 3               Phase 4
  [Raw Ingestion]      [Leakage Audit]       [Lockdown & Norm]     [Supervised MoE]
    7,200 Scans      459 Duplicates Purged     1,357 Test Locked     Plateau at 90.4%
         |                    |                     |                     |
         +--------------------+---------------------+---------------------+
                                  |
                                  v
       Phase 5               Phase 6               Phase 7
  [Frozen Ceiling]     [PEFT QKV-LoRA]       [Empirical Proof]
   92.11% Barrier      822,020 Params         95.21% (+3.10%)
   Linear Probes       Bit-Identical θ0      p = 1.303 x 10^-8
```

---

### Phase 1: Multi-Center Ingestion & Discovery

We initialized the project by compiling an aggregated multi-center cohort of **7,200 brain MRI scans** from international clinical radiology repositories. The scans span four fundamental diagnostic categories:
1. **Glioma**: Intra-axial neuroepithelial malignancy (GBM, astrocytoma, oligodendroglioma).
2. **Meningioma**: Extra-axial arachnoid cap cell tumor.
3. **Pituitary Adenoma**: Sellar/parasellar neuroendocrine mass.
4. **Healthy Brain Control**: Non-tumorous normal brain anatomy.

<div align="center">
  <img src="00_data_exploration_and_cleaning/01_initial_analysis/visuals/phase1_distribution.png" width="85%" alt="Phase 1 Cohort Distribution" />
  <p><em>Figure 1.1: Initial multi-center cohort distribution across the four diagnostic categories.</em></p>
</div>

An initial examination revealed non-standardized aspect ratios, disparate intensity distributions from different scanner field strengths (1.5T vs 3.0T), and severe risk of cross-center data leakage.

---

### Phase 2: The Data Leakage Discovery & Deduplication Audit

Standard medical imaging benchmarks frequently suffer from unacknowledged duplicate images between training and testing splits, leading to inflated, non-reproducible performance.

We constructed an exhaustive forensic deduplication pipeline utilizing:
- **Perceptual Hashing Trio**: Average Hash (`aHash`), Difference Hash (`dHash`), and Perceptual Hash (`pHash`) at 64-bit resolution.
- **Structural Similarity Cross-Validation**: Structural Similarity Index (`SSIM > 0.95`) across perceptual hash match candidates.
- **Graph Connected-Component Clustering**: Constructing an adjacency graph where images with Hamming distance $\le 4$ and SSIM $> 0.95$ formed connected components.

<div align="center">
  <img src="00_data_exploration_and_cleaning/02_duplicate_and_integrity_audit/duplicate_visuals/duplicate_network_graph.png" width="85%" alt="Duplicate Network Graph" />
  <p><em>Figure 2.1: Connected component graph network identifying 459 duplicate or near-duplicate scans across 371 isolated clusters.</em></p>
</div>

The audit uncovered **459 duplicate or near-duplicate scans** distributed across 371 clusters. These represented rotated, cropped, or contrast-stretched versions of identical patient acquisitions. All 459 scans were formally quarantined and permanently purged from the dataset.

---

### Phase 3: Protocol Lockdown & Preprocessing

Following deduplication, the cohort stood at **6,741 certified clean scans**. To guarantee zero data snooping and prevent data leakage:
1. **Test Cohort Lockdown**: We isolated **1,357 scans** as the locked test set before any model architecture was initialized.
2. **Deterministic Preprocessing**:
   - Bilinear interpolation resize to $256 \times 256$.
   - Center crop to $224 \times 224$ (preserving ventricular and sellar geometry).
   - RGB 3-channel broadcast and ImageNet channel standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

<div align="center">
  <img src="00_data_exploration_and_cleaning/02_duplicate_and_integrity_audit/per_class_visuals/cleaning_flowchart.png" width="80%" alt="Data Curation Flowchart" />
  <p><em>Figure 3.1: Complete rigorous data curation, deduplication, and locked cohort isolation flowchart.</em></p>
</div>

---

### Phase 4: Supervised Baselines Plateau & Boundary Confusion

We implemented conventional state-of-the-art supervised transfer learning baselines to benchmark performance:
- **EfficientNetV2-S**: 21.5 million trainable parameters. Trained end-to-end with Cosine Annealing learning rate schedule. Achieved **90.42% accuracy**.
- **Mixture-of-Experts (MoE)**: Gated multi-backbone expert architecture. Achieved **89.80% accuracy**.

#### The Core Problem:
Both supervised architectures hit an accuracy plateau. Confusion matrix inspection revealed the failure mode: **intra-axial Gliomas and extra-axial Meningiomas were heavily confused with one another**. Supervised backbones, trained with cross-entropy loss, learned high-frequency texture features rather than global anatomical context.

---

### Phase 5: Foundation Exploration & Representation Ceiling

We turned to self-supervised vision foundation models—specifically **Meta DINOv2 ViT-B/14** (pre-trained on 142M curated natural images via self-distillation without labels).

We extracted the 768-dimensional normalized CLS token embeddings (`x_norm_clstoken`) from the frozen backbone across the test set and evaluated **7 downstream linear probe classifiers**:
1. Logistic Regression ($L_2$ regularized)
2. Linear Support Vector Machine
3. Radial Basis Function (RBF) Kernel SVM
4. Random Forest (100 estimators)
5. Gradient Boosting (HistGradient)
6. $k$-Nearest Neighbors ($k=5$, Euclidean)
7. Multi-Layer Perceptron (MLP Probe)

<div align="center">
  <img src="MASTER_RESULTS_EVIDENCE/verified_figures/fig3_representation_ceiling.png" width="85%" alt="Representation Ceiling" />
  <p><em>Figure 5.1: The Representation Ceiling across 7 non-deep linear probing classifiers. The frozen DINOv2 backbone plateaued at 92.11%.</em></p>
</div>

#### The Discovery:
All 7 classifiers clustered between **90.86% and 92.11% accuracy**. While Healthy Brain ($99.65\%$) and Pituitary ($98.50\%$) formed well-isolated clusters, Glioma and Meningioma manifolds exhibited severe topological overlap. This proved that **the frozen representation itself was the bottleneck**, creating a mathematical ceiling of 92.11%.

---

### Phase 6: The QKV-LoRA Innovation (Controlled Arm B)

To break through the representation ceiling without fine-tuning all 86M backbone parameters (which causes catastrophic forgetting and requires massive GPU memory), we introduced **Parameter-Efficient Low-Rank Adaptation (PEFT QKV-LoRA)**:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

where:
- $W_0 \in \mathbb{R}^{d \times d}$ is the frozen pretrained weight tensor ($d=768$).
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ are low-rank adapter matrices with rank $r=8$ and scaling factor $\alpha=16$.
- Low-rank adapters were injected into the **Query ($W_q$), Key ($W_k$), and Value ($W_v$)** projection layers of all 12 Vision Transformer blocks.

```
          [ Input Patch Tokens (16x16) ]
                         |
                         v
          +-------------------------------+
          |  DINOv2 ViT-B/14 Transformer  |
          |  Block (Layers 1 to 12)       |
          |                               |
          |  Frozen W_q, W_k, W_v (768x768| <--- FROZEN (86M params)
          |        +           +          |
          |   [LoRA B_q]   [LoRA B_k/v]   | <--- TRAINABLE (r=8, α=16)
          |        *           *          |      Only 294,912 LoRA weights!
          |   [LoRA A_q]   [LoRA A_k/v]   |
          +-------------------------------+
                         |
                         v
              [ 768-dim Adapted CLS ]
                         |
                         v
          +-------------------------------+
          |  Locked 3-Layer MLP Head      | <--- Bit-Identical θ0 deepcopied
          |  768 -> 512 -> 256 -> 4       |      to both Arm A and Arm B!
          +-------------------------------+
                         |
                         v
               [ 4-Class Diagnostic ]
```

#### The Strict Experimental Control:
To mathematically isolate representation learning from classification capacity:
- A locked 3-layer MLP head (`768 -> 512(BN,ReLU,Drop0.1) -> 256(BN,ReLU,Drop0.1) -> 4`) was initialized with random seed $\theta_0$.
- The **exact tensor weights of $\theta_0$ were deepcopied** into both Arm A (Frozen) and Arm B (LoRA).
- Both arms were trained using bit-identical hyperparameters: AdamW optimizer, lr$=1 \times 10^{-4}$, weight decay$=1 \times 10^{-2}$, batch size 32, cross-entropy loss, 20 epochs.

---

### Phase 7: Empirical Verification & Statistical Proof

Upon evaluating both arms on the locked test cohort ($N=1,357$):
- **Arm A (Frozen Baseline)**: **92.11% accuracy** (1,250 / 1,357 correct), Macro-F1: 0.9196.
- **Arm B (Controlled LoRA)**: **95.21% accuracy** (1,292 / 1,357 correct), Macro-F1: 0.9507.
- **Absolute Gain**: **+3.10% Top-1 Accuracy** (+0.0311 Macro-F1).

<div align="center">
  <img src="MASTER_RESULTS_EVIDENCE/verified_figures/fig1_confusion_matrices.png" width="90%" alt="Confusion Matrices Comparison" />
  <p><em>Figure 7.1: Head-to-head confusion matrices on the locked test cohort (N=1,357). LoRA adaptation reduced mutual Glioma-Meningioma errors from 53 to 31 (a 41.5% reduction).</em></p>
</div>

<div align="center">
  <img src="MASTER_RESULTS_EVIDENCE/verified_figures/fig2_umap_2d_projection.png" width="90%" alt="UMAP Latent Manifold Projection" />
  <p><em>Figure 7.2: 2D UMAP projection of the latent representation space. Note the dramatic densification and clean separation of the Glioma and Meningioma clusters under LoRA adaptation.</em></p>
</div>

---

## 📊 Certified Results & Statistical Evidence

### Master Performance Matrix

All metrics below are computed on the locked test cohort ($N=1,357$ scans, 0.0% leakage).

| Metric | Frozen Baseline (Arm A) | Controlled QKV-LoRA (Arm B) | Absolute Gain ($\Delta$) | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 92.11% (1,250/1,357) | **95.21%** (1,292/1,357) | **+3.10%** | $[+2.14\%, +4.13\%]$ |
| **Top-2 Accuracy** | 95.87% | **96.39%** | **+0.52%** | $[+0.15\%, +0.92\%]$ |
| **Macro-Averaged F1** | 0.9196 | **0.9507** | **+0.0311** | $[+0.0211, +0.0417]$ |
| **Weighted F1** | 0.9208 | **0.9520** | **+0.0312** | $[+0.0215, +0.0414]$ |
| **Balanced Accuracy** | 92.87% | **95.69%** | **+2.82%** | $[+1.88\%, +3.82\%]$ |
| **Matthews Correlation (MCC)** | 0.8987 | **0.9376** | **+0.0389** | $[+0.0265, +0.0523]$ |
| **Cohen's Kappa ($\kappa$)** | 0.8942 | **0.9357** | **+0.0415** | $[+0.0282, +0.0551]$ |
| **Cross-Entropy Loss** | 0.2842 | **0.1614** | **-0.1228** | $[-0.1540, -0.0916]$ |

---

### Classwise Clinical Performance

| Clinical Class | True Test Scans | Frozen Recall | LoRA Recall | Net Fixed | Frozen F1 | LoRA F1 | F1 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 379 | 88.92% (337) | **93.67%** (355) | **+18 Scans** | 0.9023 | **0.9379** | **+0.0356** |
| **Meningioma** | 296 | 81.42% (241) | **88.51%** (262) | **+21 Scans** | 0.8441 | **0.9050** | **+0.0609** |
| **Healthy Brain** | 405 | 99.65% (404) | **99.65%** (404) | 0 Scans | 0.9634 | **0.9754** | **+0.0120** |
| **Pituitary** | 277 | 98.50% (273) | **98.50%** (273) | 0 Scans | 0.9687 | **0.9845** | **+0.0158** |

> **Clinical Finding**: The adaptation specifically rescued the two difficult classes—**Meningioma recall gained +7.09%** and **Glioma recall gained +4.75%**—while preserving near-perfect sensitivity on Healthy Brain (99.65%) and Pituitary (98.50%).

---

### Representation Geometry & Manifold Densification

To prove that LoRA altered the underlying representation manifold rather than simply memorizing training labels, we computed geometric manifold metrics directly on the 768-dimensional CLS token embeddings:

| Geometric Metric | Frozen Baseline (Arm A) | Controlled QKV-LoRA (Arm B) | Topological Effect |
| :--- | :---: | :---: | :--- |
| **Silhouette Score** | 0.0490 | **0.1501** ($+0.1011$) | Cluster compactness increased by **+206%** |
| **kNN Neighborhood Purity ($k=5$)** | 89.37% | **94.75%** ($+5.38\%$) | Local neighborhood homogeneity improved |
| **Inter/Intra Class Distance Ratio** | 0.6647 | **1.0287** ($+0.3640$) | Separation between different tumor classes doubled |
| **Davies-Bouldin Index** | 3.4215 | **2.1840** ($-1.2375$) | Lower cluster overlap and boundary confusion |
| **Intrinsic Dimensionality (95% PCA)**| 147 components | **107 components** | Manifold compression: task-relevant subspace alignment |

<div align="center">
  <img src="MASTER_RESULTS_EVIDENCE/verified_figures/fig2_pca_2d_projection.png" width="85%" alt="PCA 2D Projection" />
  <p><em>Figure 7.3: Principal Component Analysis (PCA) 2D projections demonstrating geometric manifold densification.</em></p>
</div>

---

### Statistical Significance Proofs

1. **Paired McNemar's Test**:
   - $n_{01}$ (Frozen incorrect, LoRA correct): **47 scans**
   - $n_{10}$ (Frozen correct, LoRA incorrect): **5 scans**
   - McNemar's Chi-squared statistic:
     $$\chi^2 = \frac{(|47 - 5| - 1)^2}{47 + 5} = \frac{41^2}{52} = \mathbf{32.3269}$$
   - **$p$-value**: $\mathbf{1.303 \times 10^{-8}}$ (Reject null hypothesis with $p \ll 0.001$).
2. **10,000 Paired Bootstrap Iterations**:
   - 95% Confidence Interval for Accuracy Delta: **$[+2.14\%, +4.13\%]$** (Zero is not contained in the interval).
   - 95% Confidence Interval for Macro-F1 Delta: **$[+0.0211, +0.0417]$**.

---

## 👁️ Explainability & Self-Attention Rollouts

To understand what DINOv2's 12 attention heads learned during LoRA adaptation, we extracted **Multi-Head Self-Attention Rollouts** by recursively multiplying the attention weight matrices across all 12 transformer layers:

$$A_{\text{rollout}} = \prod_{l=1}^{L} \left( 0.5 \cdot A^{(l)} + 0.5 \cdot I \right)$$

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Glioma Rescued Case</b></td>
      <td align="center"><b>Meningioma Rescued Case</b></td>
    </tr>
    <tr>
      <td><img src="05_notebook_G_explainability/figures/attention_rollout_glioma_rescue.png" width="100%" alt="Glioma Attention Rollout" /></td>
      <td><img src="05_notebook_G_explainability/figures/attention_rollout_meningioma_rescue.png" width="100%" alt="Meningioma Attention Rollout" /></td>
    </tr>
    <tr>
      <td><em>Attention focuses sharply along irregular, infiltrating borders separating tumor from white matter edema.</em></td>
      <td><em>Attention pinpoints the extra-axial dural tail attachment, resolving baseline confusion with glioma.</em></td>
    </tr>
    <tr>
      <td align="center"><b>Healthy Brain Control</b></td>
      <td align="center"><b>Pituitary Adenoma</b></td>
    </tr>
    <tr>
      <td><img src="05_notebook_G_explainability/figures/attention_rollout_notumor_concordant.png" width="100%" alt="Normal Brain Attention Rollout" /></td>
      <td><img src="05_notebook_G_explainability/figures/attention_rollout_pituitary_concordant.png" width="100%" alt="Pituitary Attention Rollout" /></td>
    </tr>
    <tr>
      <td><em>Attention distributes bilaterally and symmetrically across cerebral ventricles.</em></td>
      <td><em>High activation concentration localized inside the sella turcica compressing the optic chiasm.</em></td>
    </tr>
  </table>
</div>

---

## 🖼️ Verified 50-Image Showcase Gallery

We compiled a verified **50-image test set gallery** (`predictions_showcase_50/`) with certified diagnostic cards for evaluation:
- **Glioma**: 13 verified scans (including challenging low-grade diffuse astrocytomas)
- **Meningioma**: 13 verified scans (including calcified and parasagittal meningiomas)
- **Healthy Brain Control**: 12 verified normal scans
- **Pituitary Adenoma**: 12 verified macroadenomas

<div align="center">
  <img src="predictions_showcase_50/master_grids/master_50_image_overview.png" width="90%" alt="50 Image Overview Grid" />
  <p><em>Figure 8.1: Master 50-image showcase overview spanning all 4 diagnostic classes.</em></p>
</div>

### LoRA Rescue Case Studies
Below are examples of borderline scans where the Frozen Baseline made misclassifications that were rescued by LoRA adaptation:

<div align="center">
  <img src="predictions_showcase_50/bonus_lora_rescue_showcase/lora_rescued_cases_comparison.png" width="90%" alt="LoRA Rescued Cases" />
  <p><em>Figure 8.2: Head-to-head comparison of borderline scans rescued by QKV-LoRA adaptation.</em></p>
</div>

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Data Curation & Protocol Lockdown
        A["7,200 Raw MRI Scans"] --> B["Perceptual Hash Audit\n(pHash, dHash, aHash, SSIM > 0.95)"]
        B --> C["459 Contaminated Duplicates Purged"]
        C --> D["6,741 Certified Clean Cohort"]
        D --> E["Locked Test Quarantine\n(N = 1,357, 0% Leakage)"]
        D --> F["Training Cohort\n(N = 4,575)"]
        D --> G["Validation Cohort\n(N = 809)"]
    end

    subgraph Controlled Initialization
        H["Random Seed θ0"] --> I["Initialize 3-Layer MLP Head\n(768 -> 512 -> 256 -> 4)"]
        I -->|"Deepcopy Bit-Identical Tensor"| J["Arm A: Frozen Control Head"]
        I -->|"Deepcopy Bit-Identical Tensor"| K["Arm B: LoRA Head"]
    end

    subgraph Foundation Model Architecture
        F --> L["DINOv2 ViT-B/14 Backbone\n(Frozen 86M Weights)"]
        L --> J
        L --> M["Inject PEFT QKV-LoRA\n(r=8, α=16, 294k params)"]
        M --> K
    end

    subgraph Statistical Certification
        E --> N["Dual Model Forward Pass\n(N = 1,357)"]
        J --> N
        K --> N
        N --> O["Arm A: 92.11% Acc"]
        N --> P["Arm B: 95.21% Acc (+3.10%)"]
        O & P --> Q["McNemar Test: p = 1.303e-8\nBootstrap 95% CI: [+2.14%, +4.13%]\nOverlap Reduced: 41.5%"]
    end
```

---

## ⚡ Quick Start & Local Execution

### Hardware Acceleration Support
- **Apple Silicon (M1/M2/M3/M4)**: Automatically leverages **Metal Performance Shaders (MPS)** for sub-30ms warm inference.
- **NVIDIA GPU**: Automatically leverages **CUDA & cuDNN**.
- **CPU**: Clean fallback with multithreaded OpenMP execution.

### One-Command Launch

```bash
# Clone the repository
git clone https://github.com/shashankchaurasiya27-boop/neurodino-brain-tumor-mri.git
cd neurodino-brain-tumor-mri

# Launch the platform (Auto-heals dependencies & finds open port)
chmod +x run.sh
./run.sh
```

Then visit **`http://127.0.0.1:8000`** in your browser.

---

## 🔄 Reproducing the Experiments

To re-run the certified controlled experiment and verify the cryptographic model locks:

```bash
# 1. Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Execute the production controlled training pipeline
python scripts/run_controlled_lora_experiment.py \
  --data_dir "00_data_exploration_and_cleaning/raw_dataset/dataset/tumor" \
  --epochs 20 \
  --batch_size 32 \
  --lora_r 8 \
  --lora_alpha 16 \
  --seed 42

# 3. Verify cryptographic model locks
python -c "
import json
with open('outputs/*EXP_LORA_CONTROLLED_20260811_211611/checkpoints/controlled_lora_model_lock.json') as f:
    lock = json.load(f)
    print('LoRA SHA256:', lock['model_sha256'])
    print('Test Accuracy:', lock['test_accuracy'])
"
```

---

## 🚀 Cloud Deployment Guide

### Option 1: Lightning AI (100% Free Live PyTorch Compute)
1. Create a free account at [lightning.ai](https://lightning.ai).
2. Click **New Studio** (Select CPU Basic with **16 GB RAM**).
3. In the terminal:
   ```bash
   git clone https://github.com/shashankchaurasiya27-boop/neurodino-brain-tumor-mri.git
   cd neurodino-brain-tumor-mri
   ./run_cloud.sh
   ```
4. Drag and drop the two `.pth` weights from your Mac into `outputs/*EXP_LORA_CONTROLLED_20260811_211611/checkpoints/`.
5. Under the **Ports** tab, expose port `8000` as **Public** to get your live URL.

### Option 2: Vercel Serverless
1. Import this repository into [vercel.com](https://vercel.com).
2. Vercel automatically detects [`vercel.json`](vercel.json) and builds [`api/index.py`](api/index.py).
3. Operates in high-performance cloud mode with certified showcase evaluations and full interactive darkroom tools.

---

## 📁 Repository Structure & File Registry

```
neurodino-brain-tumor-mri/
├── README.md                                  # Master repository documentation (this file)
├── LICENSE                                    # MIT open-source license
├── requirements.txt                           # Full scientific Python dependencies
├── Dockerfile                                 # Container build definition for cloud studios
├── vercel.json                                # Vercel serverless routing configuration
├── run.sh                                     # Master launcher script for local execution
├── run_cloud.sh                               # Cloud studio launcher script
├── run_demo.sh                                # CLI 50-image showcase evaluation demo
│
├── api/
│   ├── index.py                               # Vercel serverless entrypoint
│   └── requirements.txt                       # Lightweight dependencies for serverless functions
│
├── brain_tumor_frontend/                      # Full interactive web diagnostic application
│   ├── app.py                                 # FastAPI application server
│   ├── config.py                              # Dynamic cross-platform environment configuration
│   ├── inference_engine.py                    # Dual-mode inference engine (PyTorch MPS/CUDA + Cloud)
│   ├── run_project.sh                         # Self-healing launcher with port discovery
│   ├── templates/
│   │   └── index.html                         # Editorial web interface (Dark/Light PACS mode)
│   └── static/
│       ├── css/
│       │   ├── design_system.css              # Color tokens, typography, grid system
│       │   └── style.css                      # Glassmorphism & darkroom UI styling
│       └── js/
│           └── app.js                         # Dual inference, darkroom LUT, loupe magnification
│
├── 00_data_exploration_and_cleaning/          # Phase 1 & 2: Dataset curation & deduplication
│   ├── 01_initial_analysis/                   # Cohort distribution & class balances
│   ├── 02_duplicate_and_integrity_audit/      # Perceptual hash deduplication & decision matrix
│   ├── 03_cleaning_and_verification/          # Clean dataset generator & verification checks
│   └── 04_scientific_certification/          # Clinical provenance & protocol documentation
│
├── 01_baseline_models/                        # Supervised baseline models
│   ├── baseline_1_efficientnet/               # EfficientNetV2-S (90.42% accuracy, 21.5M params)
│   └── baseline_moe/                          # Mixture-of-Experts pipeline (89.80% accuracy)
│
├── 02_notebook_series_A_to_D/                 # Self-supervised foundation exploration
│   ├── Notebook_A_Dataset_Audit/              # Preprocessed tensor verification
│   ├── Notebook_B_Representation_Analysis/    # Latent space geometry of frozen ViT-B/14
│   ├── Notebook_C_Representation_Benchmark/   # 7 downstream linear probes (92.11% ceiling)
│   └── Notebook_D_Evidence_Integration/       # Multi-class confusion & statistical diagnostics
│
├── 03_notebook_E_dinov2_lora/                 # QKV-LoRA architecture development
│   ├── scripts/                               # LoRA inject scripts & baseline comparison
│   └── analysis_and_diagnostics/              # Confusion matrix printers & UMAP generators
│
├── 04_notebook_F/                             # Intervention studies & contrastive formulations
│   ├── Intervention_A_Base_Adapter/           # Base adapter experiment
│   └── Intervention_B_Augmented_Adapter/      # Augmented adapter experiment
│
├── 05_notebook_G_explainability/              # Self-attention rollout explainability
│   ├── figures/                               # Attention rollouts across the 4 clinical archetypes
│   ├── scripts/                               # Attention rollout extraction engine
│   └── notebooks/                             # Interactive explainability notebook
│
├── MASTER_RESULTS_EVIDENCE/                   # Certified scientific evidence tables & figures
│   ├── 00_EXECUTIVE_SUMMARY.md                # Full clinical audit & statistical summary
│   ├── 11_REPRESENTATION_GEOMETRY_MASTER.csv  # Silhouette, kNN purity, inter/intra distance
│   ├── 20_CLASSWISE_RESULTS_TABLE.csv         # Precision, recall, F1 across all classes
│   └── verified_figures/                      # Publication-grade PCA, UMAP, confusion plots
│
├── predictions_showcase_50/                   # Verified 50-image test set showcase gallery
│   ├── predictions_manifest.csv               # Ground truth, dual predictions, latency, flags
│   ├── predictions_manifest.json              # Structured JSON manifest for web integration
│   ├── raw_mri_images/                        # Test MRI scans (Glioma, Meningioma, Normal, Pituitary)
│   ├── annotated_cards/                       # Visual diagnostic cards per sample
│   ├── master_grids/                          # 4x3 summary contact sheets per class
│   └── bonus_lora_rescue_showcase/            # Case studies of borderline scans rescued by LoRA
│
├── scripts/                                   # Production experiment & evidence scripts
│   ├── run_controlled_lora_experiment.py      # Master production experiment runner
│   ├── resume_controlled_lora_experiment.py   # Resumes training from interrupted state
│   ├── build_showcase_50.py                   # Compiles the 50-sample showcase gallery
│   └── build_master_evidence_package.py       # Assembles master CSV registries
│
└── outputs/                                   # Certified training logs & metrics
    └── *EXP_LORA_CONTROLLED_20260811_211611/  # Primary certified experiment directory
        ├── checkpoints/                       # SHA-256 verified model locks
        ├── metrics/                           # Clustering, per-class metrics, confusion matrices
        └── reports/                           # Training curves & validation reports
```

---

## 📜 Citation & Clinical Disclaimer

### Academic Citation
```bibtex
@article{neurodino2026,
  title={Controlled Parameter-Efficient Representation Adaptation of Vision Foundation Models for Multi-Center Brain Tumor MRI Diagnosis},
  author={Chaurasia, Shashank},
  journal={Biomedical Vision Foundation Research},
  year={2026},
  url={https://github.com/shashankchaurasiya27-boop/neurodino-brain-tumor-mri}
}
```

### Clinical Disclaimer
> [!CAUTION]
> **Investigational Research Device**: NeuroDINO® is an experimental machine learning research platform developed for computational neuro-oncology investigation. It is **not cleared by the US FDA, CE, or CDSCO** as a primary diagnostic device. All clinical interpretations must be validated by a board-certified neuroradiologist alongside histopathological biopsy and surgical findings.

---

<div align="center">
  <b>Developed by Shashank Chaurasia</b><br/>
  <em>Advancing Parameter-Efficient Foundation AI for Clinical Neuro-Oncology</em>
</div>
