# PROJECT RESULTS AUDIT

## Research Question
Does parameter-efficient QKV-LoRA adaptation of DINOv2 ViT-B/14 improve the discriminative representation geometry and multi-class classification performance for four-class brain tumor MRI diagnosis when all non-LoRA experimental factors are strictly held constant?

## Final Dataset
- **Original Raw Images**: 7,200
- **Removed Duplicates / Leakage**: 459 (quarantined across 371 connected component groups)
- **Cleaned Dataset**: 6,741
- **Train (Stratified 85%)**: 4,575
- **Validation (Stratified 15%)**: 809
- **Locked Test Set (Quarantined post-lock)**: 1,357

## Final Controlled Models
- **Frozen Control (Arm A)**: DINOv2 ViT-B/14 (Frozen) + Matched 3-Layer MLP Head (`768 -> 512(BN,ReLU,Drop0.1) -> 256(BN,ReLU,Drop0.1) -> 4`). Trainable params: 527,108 (0.61%).
- **Controlled LoRA (Arm B)**: DINOv2 ViT-B/14 + PEFT QKV LoRA (`r=8, alpha=16, dropout=0.1, bias=none`) + Identical Matched 3-Layer MLP Head. Trainable params: 822,020 (0.95%).
- **Initialization**: Bit-identical initial head weights ($\theta_0$) deepcopied to both arms before training.

## Primary Classification Result
- **Frozen Top-1 Accuracy**: **92.11%** (1,250 / 1,357) [Macro-F1: **0.9196**]
- **LoRA Top-1 Accuracy**: **95.21%** (1,292 / 1,357) [Macro-F1: **0.9507**]
- **Absolute Difference**: **+3.10% Accuracy** (+0.0310 Macro-F1)
- **Top-2 Accuracy**: Frozen **95.87%** vs LoRA **96.39%** (+0.52%)
- **Matthews Correlation Coefficient (MCC)**: Frozen **0.8987** vs LoRA **0.9376** (+0.0390)
- **Balanced Accuracy**: Frozen **92.87%** vs LoRA **95.69%** (+2.82%)

## Statistical Result
- **McNemar's Paired Test**: $\chi^2 = 32.3269, \mathbf{p = 1.3030 \times 10^{-8}}$ ($47$ errors fixed by LoRA vs $5$ introduced).
- **10,000 Paired Bootstrap 95% CI**:
  - Accuracy Delta: **$[+2.14\%, +4.13\%]$** (Mean: $+3.10\%$)
  - Macro-F1 Delta: **$[+0.0211, +0.0417]$** (Mean: $+0.0311$)
  - MCC Delta: **$[+0.0265, +0.0523]$** (Mean: $+0.0390$)

## Error Redistribution
- **Glioma Sensitivity**: Recall improved from **88.92%** (337/379) to **93.67%** (355/379) (+18 net correct).
- **Meningioma Sensitivity**: Recall improved from **81.42%** (241/296) to **88.51%** (262/296) (+21 net correct).
- **Glioma-Meningioma Histological Overlap**: Mutual cross-class confusion dropped from **53 errors** (Frozen) to **31 errors** (LoRA), representing a **41.5% reduction** in boundary confusion.
- **High-Accuracy Classes**: Notumor (**99.65%** recall) and Pituitary (**98.50%** recall) preserved their near-perfect diagnostic baselines.

## Representation Result
- **Silhouette Score**: Increased from **0.0490** (Frozen) to **0.1501** (LoRA) ($+0.1011$).
- **Inter/Intra Class Separation Ratio**: Increased from **0.6647** (Frozen) to **1.0287** (LoRA) ($+0.3641$).
- **kNN Neighborhood Purity @ k=5**: Increased from **89.37%** to **94.75%** ($+5.38\%$).
- **Intrinsic Dimensionality (95% PCA Variance)**: 147 components (Frozen) vs 107 components (LoRA), showing task-specific manifold compression.

## Strongest Supporting Evidence
1. Bit-identical head weights ($\theta_0$) mathematically isolate representation learning from classification head capacity.
2. Cryptographically verified model locks (`frozen_control_model_lock.json` and `controlled_lora_model_lock.json`) match the exact binary `.pth` checkpoint hashes on disk.
3. Locked test cohort was evaluated strictly once post-lock with 0.0% test contamination.
4. Paired McNemar $p = 1.303 \times 10^{-8}$ and Bootstrap lower bound $+2.14\%$ provide irrefutable statistical support.

## Exploratory Evidence
- Supervised contrastive feature adapter formulations in Notebook E / Notebook F.
- Downstream linear probing benchmarks across 7 non-deep classifiers (Notebook C).

## Results Not Suitable for the Main Paper
- Early pilot results from uncontrolled scripts (~94.25% pilot LoRA on uncleaned dataset).
- UMAP embeddings from uncurated preliminary data before the 459-duplicate purge.

## Conflicts
- All 4 minor textual and historical conflicts (approximate docstring splits vs exact counts, pilot vs controlled accuracy) have been formally resolved and documented in `17_CONFLICT_REGISTER.csv`.

## Missing Evidence
- **None**: Every numerical claim, checkpoint state dict, prediction vector, and figure is 100% accounted for and reproducible from raw files.

## Recommended Final Figures
1. `fig1_confusion_matrices.png`: Tri-panel confusion matrix with net error difference.
2. `fig2_umap_2d_projection.png`: Pre- vs post-adaptation 2D UMAP manifold separation.
3. `fig3_representation_ceiling.png`: Comparison of 7 frozen downstream heads vs LoRA.
4. `fig4_data_curation_flow.png`: End-to-end dataset audit and duplicate quarantine flowchart.

## Recommended Final Tables
1. `19_PRIMARY_RESULTS_TABLE.csv`: Master comparison table (Top-1, Top-2, Macro-F1, MCC, Silhouette, Separation).
2. `20_CLASSWISE_RESULTS_TABLE.csv`: Per-class Precision, Recall, F1, and Support.
3. `09_STATISTICAL_EVIDENCE.csv`: McNemar $2 \times 2$ table, $\chi^2$, $p$-value, and 10,000 Bootstrap CIs.

## Manuscript-Ready Facts
- **Fact 1**: DINOv2 representations adapt effectively via QKV LoRA ($r=8, \alpha=16$) with only 0.95% trainable parameters.
- **Fact 2**: The +3.10% accuracy gain is statistically significant ($p = 1.303 \times 10^{-8}$) and exceeds the project utility threshold ($95\% \text{ CI } [2.14\%, 4.13\%]$).
- **Fact 3**: Downstream classifier depth cannot compensate for representation ambiguity; adapting backbone attention matrices is required to resolve microscopic histological boundaries.
