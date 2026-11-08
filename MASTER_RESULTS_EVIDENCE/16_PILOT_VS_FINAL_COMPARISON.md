# Pilot Experiment vs. Final Primary Controlled Experiment Comparison

| Parameter / Protocol | Notebook E Pilot (`baseline_lora.py`) | Final Controlled Run (`EXP_LORA_CONTROLLED_20260811_211611`) | Scientific Impact & Rationale |
| :--- | :--- | :--- | :--- |
| **Scientific Objective** | Exploratory pilot of LoRA viability | Rigorous 2-arm hypothesis test (Frozen vs LoRA) | Eliminates all confounding non-backbone variables |
| **Dataset State** | Raw Archive (7,200 images, uncleaned) | Certified Cleaned (6,741 images, 0 duplicates) | Prevents artificial cross-split test leakage |
| **Classifier Head** | `BN(768) -> Linear(512) -> ReLU -> Drop(0.3) -> Linear(4)` | `768 -> 512(BN,ReLU,Drop0.1) -> 256(BN,ReLU,Drop0.1) -> 4` | Matched, standardized 3-layer MLP architecture |
| **Initial Head Weights** | Randomly initialized independently | Bit-Identical $\theta_0$ copied from master initialization | Guarantees differences stem strictly from backbone adaptation |
| **Data Augmentation** | Heavy Stochastic (ColorJitter, RandomResizedCrop) | Deterministic (Resize 256 -> CenterCrop 224 -> Norm) | Isolates representation learning from synthetic data shift |
| **Test Set Quarantine** | Evaluated every epoch (Contamination risk) | Strict Model Lock; Evaluated ONCE post-lock | Eliminates post-hoc cherry-picking & test snooping |
| **Reported Top-1 Accuracy** | ~94.25% | **95.21%** (Frozen Control: **92.11%**) | +3.10% true gain ($p = 1.303 \times 10^{-8}$) |
| **Publication Role** | **EXPLORATORY / OBSOLETE** | **PRIMARY CANONICAL EVIDENCE** | Only the controlled result is publication-grade |
