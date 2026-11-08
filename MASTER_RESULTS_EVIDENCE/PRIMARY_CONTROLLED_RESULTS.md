# Primary Controlled Experiment Results & Verification
**Experiment ID**: `EXP_LORA_CONTROLLED_20260811_211611`  
**Model Architecture**: Matched DINOv2 ViT-B/14 (Frozen Control vs PEFT QKV LoRA `r=8`, `alpha=16`, `drop=0.1`)  
**Classifier Head**: Bit-Identical `768 -> 512(BN,ReLU,Drop0.1) -> 256(BN,ReLU,Drop0.1) -> 4`  
**Held-out Test Cohort**: Exactly 1,357 verified MRI images (Quarantined post-lock)  

## 1. Primary Test Metrics Comparison Table

| Metric | Frozen DINOv2 (Arm A) | QKV-LoRA (Arm B) | Absolute Difference | Relative Gain | Source | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Top-1 Accuracy (%) | 92.11% (92.114959%) | 95.21% (95.210022%) | +3.10% | +3.36% | predictions/controlled_lora_predictions.csv | [RAW_VERIFIED] |
| Top-2 Accuracy (%) | 95.87% (95.873250%) | 96.39% (96.389094%) | +0.52% | +0.54% | predictions/controlled_lora_predictions.csv | [DERIVED] |
| Macro-F1 Score | 0.9196 (0.919639) | 0.9507 (0.950655) | +0.0310 | +3.37% | predictions/controlled_lora_predictions.csv | [RAW_VERIFIED] |
| Balanced Accuracy (%) | 92.87% (92.869068%) | 95.69% (95.685797%) | +2.82% | +3.03% | predictions/controlled_lora_predictions.csv | [DERIVED] |
| Matthews Correlation (MCC) | 0.8987 (0.898671) | 0.9376 (0.937635) | +0.0390 | +4.34% | predictions/controlled_lora_predictions.csv | [DERIVED] |
| Silhouette Score | 0.0490 (0.049016) | 0.1501 (0.150103) | +0.1011 | +206.23% | features/lora_test_features.npy | [DERIVED] |
| Separation Ratio | 0.6647 (0.664669) | 1.0287 (1.028738) | +0.3641 | +54.77% | features/lora_test_features.npy | [DERIVED] |

## 2. Statistical Significance Verification
* **McNemar's Chi-Squared Statistic**: $\chi^2 = 32.3269$
* **McNemar Exact $p$-value**: **$1.3030e-08$** (Vastly exceeds $p < 0.001$ threshold).
* **Paired Contingency Table ($2 \times 2$)**:
  * Both Correct ($A$): **1245**
  * Frozen Wrong $\rightarrow$ LoRA Correct ($B$): **47**
  * Frozen Correct $\rightarrow$ LoRA Wrong ($C$): **5**
  * Both Wrong ($D$): **60**
* **10,000-Iteration Paired Bootstrap 95% Confidence Intervals**:
  * **Accuracy Delta ($\%$)**: `[2.14%, 4.13%]` (Mean: `+3.10%`)
  * **Macro-F1 Delta**: `[0.0211, 0.0417]` (Mean: `+0.0311`)
  * **MCC Delta**: `[0.0265, 0.0523]` (Mean: `+0.0390`)

## 3. Per-Class Performance Table

| class | model | precision | recall | f1_score | support | source | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glioma | Frozen DINOv2 (Arm A) | 0.9863481228668942 | 0.762532981530343 | 0.8601190476190477 | 379 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/frozen_control_per_class_metrics.csv | [RAW_VERIFIED] |
| glioma | Controlled LoRA (Arm B) | 0.996875 | 0.841688654353562 | 0.9127324749642346 | 379 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/controlled_lora_per_class_metrics.csv | [RAW_VERIFIED] |
| glioma | Delta (LoRA - Frozen) | 0.010526877133105739 | 0.079155672823219 | 0.052613427345186925 | 379 | Recomputed Delta | [DERIVED] |
| meningioma | Frozen DINOv2 (Arm A) | 0.8011049723756906 | 0.9797297297297297 | 0.8814589665653495 | 296 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/frozen_control_per_class_metrics.csv | [RAW_VERIFIED] |
| meningioma | Controlled LoRA (Arm B) | 0.8963414634146342 | 0.9932432432432432 | 0.9423076923076923 | 296 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/controlled_lora_per_class_metrics.csv | [RAW_VERIFIED] |
| meningioma | Delta (LoRA - Frozen) | 0.0952364910389436 | 0.013513513513513487 | 0.06084872574234279 | 296 | Recomputed Delta | [DERIVED] |
| notumor | Frozen DINOv2 (Arm A) | 0.9245901639344263 | 1.0 | 0.9608177172061328 | 282 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/frozen_control_per_class_metrics.csv | [RAW_VERIFIED] |
| notumor | Controlled LoRA (Arm B) | 0.9185667752442996 | 1.0 | 0.9575551782682513 | 282 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/controlled_lora_per_class_metrics.csv | [RAW_VERIFIED] |
| notumor | Delta (LoRA - Frozen) | -0.006023388690126641 | 0.0 | -0.0032625389378815317 | 282 | Recomputed Delta | [DERIVED] |
| pituitary | Frozen DINOv2 (Arm A) | 0.9798488664987406 | 0.9725 | 0.9761606022584692 | 400 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/frozen_control_per_class_metrics.csv | [RAW_VERIFIED] |
| pituitary | Controlled LoRA (Arm B) | 0.9875621890547264 | 0.9925 | 0.9900249376558603 | 400 | outputs/EXP_LORA_CONTROLLED_20260811_211611/metrics/controlled_lora_per_class_metrics.csv | [RAW_VERIFIED] |
| pituitary | Delta (LoRA - Frozen) | 0.00771332255598578 | 0.020000000000000018 | 0.01386433539739107 | 400 | Recomputed Delta | [DERIVED] |

## 4. Confusion Matrices ($4 \times 4$)

### Arm A: Matched Frozen Control DINOv2 (1,250 Correct / 107 Errors)
```
                 Pred Glioma   Pred Meningioma  Pred NoTumor  Pred Pituitary  Total
True Glioma              289               61            22               7     379
True Meningioma            4              290             1               1     296
True NoTumor               0                0           282               0     282
True Pituitary             0               11             0             389     400
```

### Arm B: Controlled QKV-LoRA DINOv2 (1,292 Correct / 65 Errors)
```
                 Pred Glioma   Pred Meningioma  Pred NoTumor  Pred Pituitary  Total
True Glioma              319               31            25               4     379
True Meningioma            1              294             0               1     296
True NoTumor               0                0           282               0     282
True Pituitary             0                3             0             397     400
```

### Difference Matrix (LoRA - Frozen): Net +42 Correct Rescues
```
                 Pred Glioma   Pred Meningioma  Pred NoTumor  Pred Pituitary  Net Shift
True Glioma              +30              -30            +3              -3   +30 Rescued
True Meningioma           -3               +4            -1              +0    +4 Rescued
True NoTumor              +0               +0            +0              +0       0 Shift
True Pituitary            +0               -8            +0              +8    +8 Rescued
```
