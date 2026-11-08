# Result Claim Evidence Map (Traceability Matrix)

### Claim 1: Cleaned cohort eliminates 100% of data leakage.
* **Evidence**: NetworkX graph-partitioned duplicate quarantine moving 459 duplicate/leakage images into `duplicates/Group_XXXX/`.
* **Source**: [`00_data_exploration_and_cleaning/03_cleaning_and_verification/reports/cleaning_report.md`](file:///Users/shashankchaurasia/Documents/tumor%20copy/00_data_exploration_and_cleaning/03_cleaning_and_verification/reports/cleaning_report.md)
* **Status**: `[RAW_VERIFIED]`
* **Table**: `03_DATASET_EVIDENCE.csv`

### Claim 2: Controlled QKV-LoRA improves Top-1 classification accuracy by +3.10 percentage points over matched control.
* **Evidence**: Post-lock test set evaluation on 1,357 images yielding 95.21% vs 92.11%.
* **Source**: [`outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/master_comparison.csv`](file:///Users/shashankchaurasia/Documents/tumor%20copy/outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/master_comparison.csv)
* **Status**: `[RAW_VERIFIED]`
* **Table**: `19_PRIMARY_RESULTS_TABLE.csv`
* **Figure**: `fig1_confusion_matrices.png`

### Claim 3: The performance delta is statistically significant and not an artifact of random test partitioning.
* **Evidence**: McNemar's test $\chi^2 = 32.33, p = 1.303 \times 10^-8$ and 10,000 paired bootstrap 95% CI $[+2.14\%, +4.13\%]$.
* **Source**: [`outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/statistical_significance.json`](file:///Users/shashankchaurasia/Documents/tumor%20copy/outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/statistical_significance.json)
* **Status**: `[DERIVED]`
* **Table**: `09_STATISTICAL_EVIDENCE.csv`

### Claim 4: LoRA adaptation reshapes representation geometry rather than simply memorizing training labels.
* **Evidence**: Pre-classifier 768-D representation features exhibit a +0.1011 increase in Silhouette Score, a +5.38% increase in kNN neighborhood purity, and a separation ratio increase from 0.66 to 1.03.
* **Source**: [`outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/geometry_metrics.csv`](file:///Users/shashankchaurasia/Documents/tumor%20copy/outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/geometry_metrics.csv)
* **Status**: `[RAW_VERIFIED]`
* **Table**: `11_REPRESENTATION_GEOMETRY_MASTER.csv`
* **Figure**: `fig2_umap_2d_projection.png`
