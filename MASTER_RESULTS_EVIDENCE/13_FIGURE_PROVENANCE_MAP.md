# Figure Provenance Map (Single Source of Truth)

### Figure 1: Tri-Panel Confusion Matrix (`confusion_matrices.png`)
* **Figure Path**: `outputs/EXP_LORA_CONTROLLED_20260811_211611/figures/confusion_matrices.png`
* **Generation Script**: [`scripts/run_controlled_lora_experiment.py:1003-1021`](file:///Users/shashankchaurasia/Documents/tumor%20copy/scripts/run_controlled_lora_experiment.py#L1003-L1021)
* **Input Data**: `predictions/frozen_control_predictions.csv` & `predictions/controlled_lora_predictions.csv`
* **Experiment ID**: `EXP_LORA_CONTROLLED_20260811_211611`
* **Split**: Locked Held-Out Test Set (1,357 images)
* **Class Ordering**: `['glioma', 'meningioma', 'notumor', 'pituitary']` (Indices: 0, 1, 2, 3)
* **Scientific Value**: Visually proves the net +18 glioma and +21 meningioma error correction while holding notumor and pituitary at 99%+ accuracy.
* **Status**: **`FINAL-PAPER-CANDIDATE`**

### Figure 2: Representation Space 2D UMAP Projection (`umap_2d_projection.png`)
* **Figure Path**: `outputs/EXP_LORA_CONTROLLED_20260811_211611/figures/umap_2d_projection.png`
* **Generation Script**: [`scripts/run_controlled_lora_experiment.py:914-935`](file:///Users/shashankchaurasia/Documents/tumor%20copy/scripts/run_controlled_lora_experiment.py#L914-L935)
* **Input Data**: `features/frozen_test_features.npy` & `features/lora_test_features.npy`
* **Experiment ID**: `EXP_LORA_CONTROLLED_20260811_211611`
* **Split**: Locked Held-Out Test Set (1,357 images)
* **Parameters**: `n_neighbors=15, min_dist=0.1, random_state=42`
* **Scientific Value**: Shows the dramatic intra-class densification and clean geometric separation of glioma from meningioma post-LoRA adaptation.
* **Status**: **`FINAL-PAPER-CANDIDATE`**

### Figure 3: Linear vs LoRA Downstream Representation Ceiling (`rep_ceiling_comparison.png`)
* **Figure Path**: `MASTER_RESULTS_EVIDENCE/verified_figures/fig3_representation_ceiling.png`
* **Generation Script**: `MASTER_RESULTS_EVIDENCE/figure_scripts/generate_fig3_representation_ceiling.py`
* **Input Data**: Notebook C Linear Benchmark vs Primary Controlled Results
* **Scientific Value**: Illustrates that complex downstream classifiers on frozen embeddings plateau at 93.52%, while PEFT LoRA in the backbone breaks through to 95.21%.
* **Status**: **`FINAL-PAPER-CANDIDATE`**
