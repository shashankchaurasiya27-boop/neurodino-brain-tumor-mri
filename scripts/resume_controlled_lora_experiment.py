#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from run_controlled_lora_experiment import (
    ControlledLoRAExperimentPipeline,
    get_mri_transforms,
    MRIDataset,
    SEED,
    compute_sha256
)

def main():
    parser = argparse.ArgumentParser(description="Resume Controlled LoRA Experiment Pipeline")
    parser.add_argument("--exp-id", type=str, default="EXP_LORA_CONTROLLED_20260811_211611", help="Experiment ID to resume")
    args = parser.parse_args()

    pipeline = ControlledLoRAExperimentPipeline(dry_run=False)
    
    # Override paths to point to the existing experiment
    pipeline.exp_id = args.exp_id
    pipeline.output_dir = Path("outputs") / pipeline.exp_id
    pipeline.subdirs = {
        "configs": pipeline.output_dir / "configs",
        "checkpoints": pipeline.output_dir / "checkpoints",
        "features": pipeline.output_dir / "features",
        "metrics": pipeline.output_dir / "metrics",
        "predictions": pipeline.output_dir / "predictions",
        "figures": pipeline.output_dir / "figures",
        "reports": pipeline.output_dir / "reports",
        "logs": pipeline.output_dir / "logs"
    }
    print(f"Resuming evaluation for EXP_ID: {pipeline.exp_id}")

    dataset_base = Path("00_data_exploration_and_cleaning/data_archives/dataset_archive2/archive")
    train_samples, val_samples, test_samples = pipeline.phase_3_stratified_split(dataset_base)
    
    mri_transform = get_mri_transforms()
    train_ds = MRIDataset(train_samples, transform=mri_transform)
    val_ds = MRIDataset(val_samples, transform=mri_transform)
    test_ds = MRIDataset(test_samples, transform=mri_transform)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)
    
    arm_a, arm_b = pipeline.phase_4_to_8_models_and_audit()
    
    # Hardcode the checkpoints and hashes
    ckpt_a = pipeline.subdirs["checkpoints"] / "frozen_control_best.pth"
    ckpt_b = pipeline.subdirs["checkpoints"] / "controlled_lora_best.pth"
    
    hash_a = compute_sha256(ckpt_a)
    hash_b = compute_sha256(ckpt_b)

    # Evaluate Locked Test Set
    res_eval_a = pipeline.evaluate_locked_test(arm_a, ckpt_a, "Frozen_Control", test_loader)
    res_eval_b = pipeline.evaluate_locked_test(arm_b, ckpt_b, "Controlled_LoRA", test_loader)
    
    # Features & Geometry Analysis
    geom_res = pipeline.phase_16_to_26_features_and_geometry(arm_a, arm_b, train_loader, val_loader, test_loader)
    
    # Statistical Testing & Paired Error Analysis
    stat_res = pipeline.phase_27_to_30_statistical_testing(res_eval_a, res_eval_b)
    
    # Master Certification & Report
    pipeline.phase_33_to_41_certification_and_report(res_eval_a, res_eval_b, geom_res, stat_res, hash_a, hash_b)
    print("Post-training evaluation and report generation completed successfully.")

if __name__ == "__main__":
    main()
