"""
Configuration and cross-project linkage for Brain Tumor MRI Diagnostic Frontend.
Points cleanly to the research project at /Users/shashankchaurasia/Documents/tumor copy
without modifying any files in the original project.
"""

import os
import glob
from pathlib import Path

# Base Paths
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", FRONTEND_DIR.parent))
if not PROJECT_ROOT.exists() or not (PROJECT_ROOT / "outputs").exists():
    fallback_path = Path("/Users/shashankchaurasia/Documents/tumor copy")
    if fallback_path.exists():
        PROJECT_ROOT = fallback_path

# Checkpoint & Experiment Directory
CERTIFIED_EXP_DIR = None
potential_dirs = glob.glob(str(PROJECT_ROOT / "outputs/*20260811_211611")) + [
    str(PROJECT_ROOT / "outputs/EXP_LORA_CONTROLLED_20260811_211611")
]
for p in potential_dirs:
    if os.path.isdir(p):
        CERTIFIED_EXP_DIR = Path(p)
        break

if not CERTIFIED_EXP_DIR or not CERTIFIED_EXP_DIR.exists():
    CERTIFIED_EXP_DIR = PROJECT_ROOT / "outputs/*EXP_LORA_CONTROLLED_20260811_211611"

CHECKPOINTS_DIR = CERTIFIED_EXP_DIR / "checkpoints"
LORA_CHECKPOINT_PATH = CHECKPOINTS_DIR / "controlled_lora_best.pth"
FROZEN_CHECKPOINT_PATH = CHECKPOINTS_DIR / "frozen_control_best.pth"

# Model Locks
LORA_LOCK_PATH = CHECKPOINTS_DIR / "controlled_lora_model_lock.json"
FROZEN_LOCK_PATH = CHECKPOINTS_DIR / "frozen_control_model_lock.json"

EXPECTED_LORA_SHA256 = "59dafadda3f983952ae295d257481ad6b582112869d3fe291fbcb54a5d7dd351"
EXPECTED_FROZEN_SHA256 = "2b0206ba5f62b1d33ba0de14d9e0eac7268617999b8a735894fcae41f9cbb04c"

# Showcase & Manifests
SHOWCASE_DIR = PROJECT_ROOT / "predictions_showcase_50"
if not SHOWCASE_DIR.exists():
    SHOWCASE_DIR = FRONTEND_DIR / "predictions_showcase_50"

SHOWCASE_MANIFEST_CSV = SHOWCASE_DIR / "predictions_manifest.csv"
if not SHOWCASE_MANIFEST_CSV.exists() and (FRONTEND_DIR / "predictions_manifest.csv").exists():
    SHOWCASE_MANIFEST_CSV = FRONTEND_DIR / "predictions_manifest.csv"

SHOWCASE_MANIFEST_JSON = SHOWCASE_DIR / "predictions_manifest.json"
if not SHOWCASE_MANIFEST_JSON.exists() and (FRONTEND_DIR / "predictions_manifest.json").exists():
    SHOWCASE_MANIFEST_JSON = FRONTEND_DIR / "predictions_manifest.json"

SHOWCASE_IMAGES_DIR = SHOWCASE_DIR / "raw_mri_images"
SHOWCASE_CARDS_DIR = SHOWCASE_DIR / "annotated_cards"
SHOWCASE_GRIDS_DIR = SHOWCASE_DIR / "master_grids"
BONUS_RESCUE_DIR = SHOWCASE_DIR / "bonus_lora_rescue_showcase"

# Evidence & Verified Figures
EVIDENCE_DIR = PROJECT_ROOT / "MASTER_RESULTS_EVIDENCE"
if not EVIDENCE_DIR.exists():
    EVIDENCE_DIR = FRONTEND_DIR / "MASTER_RESULTS_EVIDENCE"

EXECUTIVE_SUMMARY_MD = EVIDENCE_DIR / "00_EXECUTIVE_SUMMARY.md"
EVIDENCE_FIGURES_DIR = EVIDENCE_DIR / "verified_figures"

REPRESENTATION_GEOMETRY_CSV = EVIDENCE_DIR / "11_REPRESENTATION_GEOMETRY_MASTER.csv"
if not REPRESENTATION_GEOMETRY_CSV.exists() and (FRONTEND_DIR / "11_REPRESENTATION_GEOMETRY_MASTER.csv").exists():
    REPRESENTATION_GEOMETRY_CSV = FRONTEND_DIR / "11_REPRESENTATION_GEOMETRY_MASTER.csv"

CLASSWISE_RESULTS_CSV = EVIDENCE_DIR / "20_CLASSWISE_RESULTS_TABLE.csv"
if not CLASSWISE_RESULTS_CSV.exists() and (FRONTEND_DIR / "20_CLASSWISE_RESULTS_TABLE.csv").exists():
    CLASSWISE_RESULTS_CSV = FRONTEND_DIR / "20_CLASSWISE_RESULTS_TABLE.csv"

# Curation & Explainability Directories
CURATION_DIR = PROJECT_ROOT / "00_data_exploration_and_cleaning"
EXPLAINABILITY_DIR = PROJECT_ROOT / "05_notebook_G_explainability/figures"

# Diagnostic Classes and Display Properties
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

CLASS_DISPLAY = {
    "glioma": {
        "title": "Glioma Tumor",
        "category": "Primary Intra-Axial Brain Tumor",
        "color": "#EC4899",     # Pink / Rose
        "is_tumor": True,
        "description": "Glial cell malignancy arising from glial tissue within the cerebral hemispheres, requiring aggressive resection."
    },
    "meningioma": {
        "title": "Meningioma Tumor",
        "category": "Extra-Axial Meningeal Tumor",
        "color": "#F59E0B",     # Amber / Gold
        "is_tumor": True,
        "description": "Typically benign, dural-based extra-axial tumor arising from arachnoid cap cells of the meninges."
    },
    "notumor": {
        "title": "Healthy Brain (No Tumor)",
        "category": "Normal Neuroanatomy Control",
        "color": "#29AE29",     # Forest / Vivid Green (Alert Success)
        "is_tumor": False,
        "description": "Normal cerebral cortex, ventricles, and white matter with zero focal intra-axial or extra-axial mass lesions."
    },
    "pituitary": {
        "title": "Pituitary Adenoma",
        "category": "Sellar / Neuroendocrine Tumor",
        "color": "#9B43F3",     # Electric Violet (Primary)
        "is_tumor": True,
        "description": "Sellar neuroendocrine mass originating from anterior pituitary gland cells, often compressing optic chiasm."
    }
}

# Certified Benchmark Metrics
BENCHMARK_METRICS = {
    "test_set_size": 1357,
    "frozen_accuracy": 92.11,
    "frozen_macro_f1": 0.9196,
    "lora_accuracy": 95.21,
    "lora_macro_f1": 0.9507,
    "accuracy_gain": 3.10,
    "mcnemar_chi2": 32.3269,
    "mcnemar_p_value": "1.303e-8",
    "bootstrap_ci_95": [2.14, 4.13],
    "glioma_meningioma_confusion_reduction": "41.5%",
    "silhouette_score_gain": "+0.1011 (0.0490 -> 0.1501)",
    "knn_purity_gain": "+5.38% (89.37% -> 94.75%)",
    "trainable_params_lora": 822020,
    "trainable_params_pct": 0.95,
    "total_params": 86527108
}

# Server Config
DEFAULT_HOST = os.environ.get("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
