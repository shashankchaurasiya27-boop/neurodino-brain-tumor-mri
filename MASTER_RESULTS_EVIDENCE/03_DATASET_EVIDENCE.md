# Dataset Evidence & Curation Summary

This document provides the verified quantitative breakdown of the dataset curation, duplicate quarantine, and split allocation.

## 1. Summary Metrics

| metric | value | source_file | location | status | calculation |
| --- | --- | --- | --- | --- | --- |
| Original Raw Image Count | 7200 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/reports/cleaning_report.md | Line 9 | [RAW_VERIFIED] | Pre-cleaning physical disk enumeration |
| Cleaned Dataset Image Count | 6741 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/reports/cleaning_report.md | Line 10 | [RAW_VERIFIED] | 7200 - 459 = 6741 |
| Removed / Quarantined Duplicates Count | 459 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | Sum of 'Number Removed' column | [RAW_VERIFIED] | Sum across 371 duplicate groups |
| Duplicate Groups Resolved | 371 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | Row count | [RAW_VERIFIED] | NetworkX connected components |
| Exact Duplicates Removed | 169 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | df_moved['reason'] == 'Exact Duplicate' | [DERIVED] | 169 exact byte/hash duplicate images |
| Critical Cross-Split Leakage Removed | 148 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | df_moved['reason'] == 'Critical Leakage' | [DERIVED] | 148 images bridging Train and Test splits |
| Pixel-Identical Variants Removed | 142 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | df_moved['reason'] == 'Pixel Identical' | [DERIVED] | 142 images with identical pixels but different compression bytes |
| Training Set Moved Images | 216 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | df_moved['split'] == 'Training' | [DERIVED] | 216 images moved from original 5,600 training |
| Testing Set Moved Images | 243 | 00_data_exploration_and_cleaning/03_cleaning_and_verification/cleaning_summary.csv | df_moved['split'] == 'Testing' | [DERIVED] | 243 images moved from original 1,600 testing |
| Final Controlled Train Split Count | 4575 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/split_manifest.csv | df_split['split'] == 'train' | [RAW_VERIFIED] | 4575 images (85% stratified train) |
| Final Controlled Validation Split Count | 809 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/split_manifest.csv | df_split['split'] == 'val' | [RAW_VERIFIED] | 809 images (15% stratified val via math.ceil per class) |
| Final Controlled Locked Test Split Count | 1357 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/split_manifest.csv | df_split['split'] == 'test' | [RAW_VERIFIED] | 1357 held-out locked test images |
| Final Total Images (Train+Val+Test) | 6741 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/split_manifest.csv | Row count | [RAW_VERIFIED] | 4575 + 809 + 1357 = 6741 |
| Dominant Image Matrix Resolution | 512x512 (73.7% of cohort) | outputs/EXP_20260804_150257/configs/dataset_manifest.csv | width x height value counts | [RAW_VERIFIED] | 4,970 / 6,741 images are 512x512 |
| Color Mode Distribution | RGB: 3682, L: 3055, RGBA: 3, P: 1 | outputs/EXP_20260804_150257/configs/dataset_manifest.csv | mode value counts | [RAW_VERIFIED] | All 8-bit intensity depth |
| Primary MRI Sequence Modality | Axial T1-weighted Contrast-Enhanced (T1-CE / T1c) | 02_notebook_series_A_to_D/Notebook_D_Evidence_Integration/baseline_d_evidence_engine.py | Line 430 | [CODE_VERIFIED] | Multi-center Figshare/Cheng/Sartaj diagnostic cohort |
| Split Random Seed | 42 | scripts/run_controlled_lora_experiment.py | Line 80 | [CODE_VERIFIED] | Locked deterministic pseudo-random seed |

## 2. Per-Class Distribution in Cleaned Cohort (6,741 images)

| Index | glioma | meningioma | notumor | pituitary | All |
| --- | --- | --- | --- | --- | --- |
| test | 379 | 296 | 282 | 400 | 1357 |
| train | 1190 | 1176 | 1055 | 1154 | 4575 |
| val | 210 | 208 | 187 | 204 | 809 |
| All | 1779 | 1680 | 1524 | 1758 | 6741 |

## 3. Duplicate Breakdown by Class & Reason (459 moved images)

| Index | glioma | meningioma | notumor | pituitary | All |
| --- | --- | --- | --- | --- | --- |
| Critical Leakage | 0 | 7 | 141 | 0 | 148 |
| Exact Duplicate | 13 | 10 | 107 | 39 | 169 |
| Pixel Identical | 8 | 103 | 28 | 3 | 142 |
| All | 21 | 120 | 276 | 42 | 459 |

