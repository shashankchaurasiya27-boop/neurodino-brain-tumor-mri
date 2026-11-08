#!/usr/bin/env python3
"""
=============================================================================
REVISED EVIDENCE CALCULATIONS SCRIPT
=============================================================================
Purpose:
  Provides the corrected, executable analytical calculations originally
  transcribed into the header of `evidence.md`. Fixes the syntax typo in
  Top-2 accuracy evaluation (replacing `np.argsort[...](-2:)` with
  `np.argsort(...)[-2:]`).

Required Input Files:
  1. outputs/EXP_LORA_CONTROLLED_20260811_211611/predictions/frozen_control_predictions.csv
  2. outputs/EXP_LORA_CONTROLLED_20260811_211611/predictions/controlled_lora_predictions.csv

Document Provenance:
  Historical Record: evidence.md (Pre-publication research log)
  Revised Calculation: scripts/revised_evidence_calculations.py
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
)
import scipy.stats as stats

def main():
    # Resolve repository paths
    base_dir = Path(__file__).resolve().parent.parent
    pred_dir = base_dir / "outputs/EXP_LORA_CONTROLLED_20260811_211611/predictions"
    
    file_f = pred_dir / "frozen_control_predictions.csv"
    file_l = pred_dir / "controlled_lora_predictions.csv"
    
    if not file_f.exists() or not file_l.exists():
        print(f"[!] Error: Missing required input files in {pred_dir}")
        sys.exit(1)
        
    print(f"[*] Loading prediction artifacts from: {pred_dir}...")
    df_pf = pd.read_csv(file_f)
    df_pl = pd.read_csv(file_l)
    
    y_true = df_pf['true_label'].values
    y_pred_f = df_pf['predicted_label'].values
    y_pred_l = df_pl['predicted_label'].values
    n_samples = len(y_true)
    
    print("\n--- 1. RAW TOP-1 ACCURACY EVALUATION ---")
    print(f"Total Evaluated Samples: {n_samples}")
    acc_f = np.mean(y_pred_f == y_true) * 100
    acc_l = np.mean(y_pred_l == y_true) * 100
    print(f"Frozen Correct: {np.sum(y_pred_f == y_true)} / {n_samples} ({acc_f:.6f}%)")
    print(f"LoRA Correct:   {np.sum(y_pred_l == y_true)} / {n_samples} ({acc_l:.6f}%)")
    print(f"Absolute Delta: {acc_l - acc_f:+.6f}%")
    
    print("\n--- 2. CORRECTED TOP-2 ACCURACY EVALUATION ---")
    probs_f = df_pf[['prob_glioma', 'prob_meningioma', 'prob_notumor', 'prob_pituitary']].values
    probs_l = df_pl[['prob_glioma', 'prob_meningioma', 'prob_notumor', 'prob_pituitary']].values
    
    # Corrected syntax: np.argsort(probs)[-2:]
    top2_f = sum([y_true[i] in np.argsort(probs_f[i])[-2:] for i in range(n_samples)])
    top2_l = sum([y_true[i] in np.argsort(probs_l[i])[-2:] for i in range(n_samples)])
    print(f"Top-2 Frozen: {top2_f} / {n_samples} ({top2_f / n_samples * 100:.6f}%)")
    print(f"Top-2 LoRA:   {top2_l} / {n_samples} ({top2_l / n_samples * 100:.6f}%)")
    
    print("\n--- 3. EMPIRICAL CONFUSION MATRICES ---")
    cm_f = confusion_matrix(y_true, y_pred_f, labels=[0, 1, 2, 3])
    cm_l = confusion_matrix(y_true, y_pred_l, labels=[0, 1, 2, 3])
    print("Frozen Control Matrix:\n", cm_f)
    print("Controlled LoRA Matrix:\n", cm_l)
    print("Net Difference (LoRA - Frozen):\n", cm_l - cm_f)
    
    print("\n--- 4. PER-CLASS PRECISION, RECALL & F1 ---")
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    for c_idx, c_name in enumerate(classes):
        prec_f = precision_score(y_true == c_idx, y_pred_f == c_idx)
        rec_f = recall_score(y_true == c_idx, y_pred_f == c_idx)
        f1_f = f1_score(y_true == c_idx, y_pred_f == c_idx)
        
        prec_l = precision_score(y_true == c_idx, y_pred_l == c_idx)
        rec_l = recall_score(y_true == c_idx, y_pred_l == c_idx)
        f1_l = f1_score(y_true == c_idx, y_pred_l == c_idx)
        
        supp = int(np.sum(y_true == c_idx))
        print(f"[{c_name.upper()}] (Support: {supp})")
        print(f"  Frozen: Prec={prec_f:.6f}, Rec={rec_f:.6f}, F1={f1_f:.6f}")
        print(f"  LoRA:   Prec={prec_l:.6f}, Rec={rec_l:.6f}, F1={f1_l:.6f}")
        print(f"  Delta:  ΔPrec={prec_l - prec_f:+.6f}, ΔRec={rec_l - rec_f:+.6f}, ΔF1={f1_l - f1_f:+.6f}")
        
    print("\n--- 5. MCNEMAR CONTINGENCY TEST ---")
    corr_f = (y_pred_f == y_true)
    corr_l = (y_pred_l == y_true)
    n11 = int(np.sum(corr_f & corr_l))
    n01 = int(np.sum(~corr_f & corr_l))
    n10 = int(np.sum(corr_f & ~corr_l))
    n00 = int(np.sum(~corr_f & ~corr_l))
    
    print(f"2x2 Contingency: Both Correct (n11)={n11}, Rescued by LoRA (n01)={n01}, Broken by LoRA (n10)={n10}, Both Wrong (n00)={n00}")
    stat_std = (n01 - n10)**2 / (n01 + n10)
    p_std = stats.chi2.sf(stat_std, 1)
    stat_edwards = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_edwards = stats.chi2.sf(stat_edwards, 1)
    print(f"Standard McNemar:  chi2={stat_std:.6f}, p={p_std:.4e}")
    print(f"Edwards Corrected: chi2={stat_edwards:.6f}, p={p_edwards:.4e}")
    print("\n[✓] All calculations executed successfully without typographical errors.")

if __name__ == "__main__":
    main()
