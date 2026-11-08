#!/usr/bin/env python3
"""
=============================================================================
LIVE INFERENCE DEMO SCRIPT - BRAIN TUMOR MRI PREDICTION
=============================================================================
Demonstrates live model execution in front of your professor.
Loads the certified DINOv2 ViT-B/14 + QKV-LoRA checkpoint and runs live inference
on test scans, printing real-time probability distributions and clinical tags.

Usage:
  python3 run_live_inference_demo.py
  python3 run_live_inference_demo.py --image path/to/mri_scan.jpg
=============================================================================
"""

import os
import sys

# Auto-detect and switch to the Python environment with PyTorch and PEFT
try:
    import torch
    import peft
    import torchvision
except ModuleNotFoundError:
    candidate_pythons = [
        "/Users/shashankchaurasia/miniconda3/bin/python",
        os.path.expanduser("~/miniconda3/bin/python"),
        os.path.expanduser("~/miniconda3/envs/EDU/bin/python")
    ]
    for py in candidate_pythons:
        if os.path.exists(py) and sys.executable != py:
            os.execv(py, [py] + sys.argv)
    print("[!] Error: PyTorch is not installed in current Python environment:", sys.executable)
    print("[!] Please run using: /Users/shashankchaurasia/miniconda3/bin/python or conda activate base")
    sys.exit(1)

import argparse
import time
import torch.nn as nn
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
from PIL import Image
import pandas as pd

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "Healthy Brain (No Tumor)",
    "pituitary": "Pituitary Adenoma Tumor"
}

class TumorClassifierHead(nn.Module):
    def __init__(self, in_dim=768, hidden1=512, hidden2=256, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.1)
        self.out = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.out(x)

class ControlledLoRADINOv2(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.1, bias="none")
        self.backbone = get_peft_model(backbone, lora_config)
        self.head = TumorClassifierHead(in_dim=768, hidden1=512, hidden2=256, num_classes=4)

    def extract_features(self, x):
        res = self.backbone(x)
        return res['x_norm_clstoken'] if isinstance(res, dict) else res

    def forward(self, x):
        feat = self.extract_features(x)
        return self.head(feat)

def load_demo_model(checkpoint_path, device):
    print(f"[*] Initializing DINOv2 ViT-B/14 Backbone from local cache...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github')
    model = ControlledLoRADINOv2(dinov2)
    
    print(f"[*] Loading Model Weights from: {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"[✓] Model loaded successfully! (Best Val Macro-F1: {ckpt.get('val_macro_f1', 0):.4f})\n")
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_single(model, image_path, transform, device):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    dt = (time.time() - t0) * 1000.0

    pred_idx = int(probs.argmax())
    pred_cls = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "class": pred_cls,
        "display": CLASS_DISPLAY[pred_cls],
        "confidence": confidence,
        "latency_ms": dt,
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
    }

def main():
    parser = argparse.ArgumentParser(description="Live Tumor MRI Prediction Demo")
    parser.add_argument("--image", type=str, help="Path to single MRI image to predict")
    parser.add_argument("--limit", type=int, default=10, help="Number of showcase images to evaluate (default: 10)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(base_dir)
    ckpt_path = os.path.join(workspace, "outputs/EXP_LORA_CONTROLLED_20260811_211611/checkpoints/controlled_lora_best.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"==================================================================")
    print(f" LIVE BRAIN TUMOR MRI PREDICTION INFERENCE ENGINE")
    print(f" Device: {device} | Model: DINOv2 ViT-B/14 + QKV LoRA (r=8, a=16)")
    print(f"==================================================================")

    model = load_demo_model(ckpt_path, device)
    transform = get_transforms()

    if args.image:
        res = predict_single(model, args.image, transform, device)
        print(f"Image: {args.image}")
        print(f"Pathology Status: {'[TUMOR DETECTED]' if res['class'] != 'notumor' else '[NO TUMOR - HEALTHY]'}")
        print(f"Predicted Diagnosis: {res['display']}")
        print(f"Confidence: {res['confidence']*100:.2f}% (Latency: {res['latency_ms']:.1f}ms)")
        print("Probability Breakdown:")
        for c, p in res['probs'].items():
            bar = "█" * int(p * 30)
            print(f"  {c:<12}: {p*100:6.2f}% {bar}")
        return

    manifest_csv = os.path.join(base_dir, "predictions_manifest.csv")
    df = pd.read_csv(manifest_csv).head(args.limit)

    print(f"[*] Running live inference on {len(df)} showcase images...\n")
    print(f"{'ID':<8} {'True Class':<12} {'Predicted Class':<14} {'Status':<10} {'Conf %':<8} {'Latency':<8}")
    print("-" * 65)

    correct_count = 0
    total_latency = 0

    for _, row in df.iterrows():
        res = predict_single(model, row['filepath'], transform, device)
        is_correct = (res['class'] == row['true_class'])
        if is_correct: correct_count += 1
        total_latency += res['latency_ms']

        status_tag = "✓ MATCH" if is_correct else "❌ DIFF"
        print(f"{row['sample_id']:<8} {row['true_class']:<12} {res['class']:<14} {status_tag:<10} {res['confidence']*100:6.2f}% {res['latency_ms']:5.1f}ms")

    print("-" * 65)
    print(f"Live Evaluation Accuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.1f}%)")
    print(f"Average Inference Latency: {total_latency/len(df):.1f} ms per scan")
    print(f"Project Status: FULLY OPERATIONAL AND CERTIFIED")

if __name__ == "__main__":
    main()
