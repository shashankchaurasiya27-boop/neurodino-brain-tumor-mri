#!/usr/bin/env python3
"""
=============================================================================
DINOv2 ViT + LoRA EXPLAINABILITY & ATTENTION ROLLOUT ENGINE
=============================================================================
Provides production-grade explainability for Vision Transformers (DINOv2 ViT-B/14)
adapted with Parameter-Efficient Fine-Tuning (PEFT QKV-LoRA).

Features:
1. Full 12-layer recursive Attention Rollout (Abnar & Zuidema, 2020) with identity
   residual preservation: A_rollout = prod_{l=1}^L (0.5 * W_l + 0.5 * I).
2. Native support for PEFT LoRA projection layers on QKV attention matrices.
3. Comparative Attention Auditing: Side-by-side evaluation of Matched Frozen Control
   (Arm A) vs Controlled QKV-LoRA (Arm B).
4. Attention differential mapping (Delta = LoRA - Frozen) highlighting rescued lesions.
5. Quantitative Sparsity and Focus Metrics (Gini Index, Spatial Entropy).
=============================================================================
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor (Healthy)",
    "pituitary": "Pituitary Adenoma"
}

# -----------------------------------------------------------------------------
# ARCHITECTURE DEFINITIONS
# -----------------------------------------------------------------------------
class TumorClassifierHead(nn.Module):
    """Locked 3-layer MLP classifier head matching certified experiment."""
    def __init__(self, in_dim: int = 768, hidden1: int = 512, hidden2: int = 256, num_classes: int = 4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.out(x)

class MatchedFrozenDINOv2(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = TumorClassifierHead(in_dim=768, hidden1=512, hidden2=256, num_classes=4)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        res = self.backbone(x)
        return res['x_norm_clstoken'] if isinstance(res, dict) else res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.head(feat)

class ControlledLoRADINOv2(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone = get_peft_model(backbone, lora_config)
        self.head = TumorClassifierHead(in_dim=768, hidden1=512, hidden2=256, num_classes=4)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        res = self.backbone(x)
        return res['x_norm_clstoken'] if isinstance(res, dict) else res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.head(feat)

# -----------------------------------------------------------------------------
# ATTENTION EXTRACTION & ROLLOUT
# -----------------------------------------------------------------------------
class ViTAttentionRollout:
    """
    Hooks into all transformer blocks of DINOv2 ViT-B/14 and computes
    recursive Attention Rollout across all 12 layers.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.attention_matrices = []
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Locate ViT blocks
        blocks = None
        if hasattr(self.model, "backbone"):
            if hasattr(self.model.backbone, "base_model"):
                # PEFT LoRA model
                blocks = self.model.backbone.base_model.model.blocks
            elif hasattr(self.model.backbone, "blocks"):
                # Standard DINOv2
                blocks = self.model.backbone.blocks
        elif hasattr(self.model, "blocks"):
            blocks = self.model.blocks

        if blocks is None:
            raise ValueError("Could not locate DINOv2 transformer blocks in model.")

        for i, block in enumerate(blocks):
            def make_hook(layer_idx):
                def hook(module, inp, out):
                    # In DINOv2 MemEffAttention, qkv output shape is (B, N, 3*C)
                    B, N, C_total = out.shape
                    C = C_total // 3
                    num_heads = 12
                    head_dim = C // num_heads
                    
                    qkv = out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)
                    
                    scale = head_dim ** -0.5
                    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                    attn = torch.softmax(attn, dim=-1)
                    self.attention_matrices.append(attn.detach())
                return hook

            h = block.attn.qkv.register_forward_hook(make_hook(i))
            self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def compute_rollout(self, tensor_img: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Executes forward pass, intercepts attention across all 12 layers,
        and computes recursive Attention Rollout.
        Returns:
            rollout_map: 224x224 float array in [0, 1]
            last_layer_map: 224x224 float array in [0, 1]
            logits: Output classification logits
        """
        self.attention_matrices.clear()
        
        with torch.no_grad():
            logits = self.model(tensor_img)

        # attention_matrices has 12 entries, each of shape (1, 12, 257, 257)
        num_layers = len(self.attention_matrices)
        if num_layers == 0:
            raise RuntimeError("No attention matrices captured by hooks.")

        # Compute Attention Rollout
        # Rollout starts at identity
        B, H, N, _ = self.attention_matrices[0].shape
        eye = torch.eye(N, device=self.device).unsqueeze(0) # (1, N, N)
        
        rollout = eye.clone()
        for l in range(num_layers):
            # Average attention across all 12 heads
            attn_layer = self.attention_matrices[l].mean(dim=1) # (1, N, N)
            
            # Add identity matrix to model residual skip connections
            attn_with_residual = 0.5 * attn_layer + 0.5 * eye
            
            # Normalize rows to sum to 1
            attn_norm = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
            
            # Recursive matrix multiplication
            rollout = torch.bmm(attn_norm, rollout)

        # Extract CLS token attention to the 256 spatial patches
        cls_rollout = rollout[0, 0, 1:].cpu().numpy() # (256,)
        
        # Last layer attention for comparison
        last_attn = self.attention_matrices[-1].mean(dim=1)[0, 0, 1:].cpu().numpy() # (256,)

        def reshape_and_upsample(vec):
            grid_size = int(np.sqrt(len(vec))) # 16
            grid = vec.reshape(grid_size, grid_size)
            # Min-Max normalize
            grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
            # Upsample to 224x224
            upsampled = cv2.resize(grid, (224, 224), interpolation=cv2.INTER_CUBIC)
            upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
            return upsampled

        rollout_map = reshape_and_upsample(cls_rollout)
        last_layer_map = reshape_and_upsample(last_attn)
        
        return rollout_map, last_layer_map, logits

# -----------------------------------------------------------------------------
# QUANTITATIVE METRICS
# -----------------------------------------------------------------------------
def compute_gini_index(array: np.ndarray) -> float:
    """Computes Gini sparsity coefficient of spatial attention map."""
    arr = np.sort(array.flatten().astype(np.float64))
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr) + 1e-12))

def compute_spatial_entropy(array: np.ndarray) -> float:
    """Computes Shannon entropy of normalized spatial attention distribution."""
    p = array.flatten().astype(np.float64)
    p = p / (np.sum(p) + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-12)))

# -----------------------------------------------------------------------------
# COMPARATIVE AUDITOR
# -----------------------------------------------------------------------------
class ComparativeAttentionAuditor:
    """
    Executes simultaneous comparative attention auditing for both Arm A (Frozen)
    and Arm B (LoRA), producing 4-panel visual evidence and quantitative metrics.
    """
    def __init__(self, frozen_ckpt: str, lora_ckpt: str, device: torch.device):
        self.device = device
        print(f"[*] Initializing DINOv2 ViT-B/14 Backbone on {device}...")
        
        backbone_frozen = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github')
        self.model_frozen = MatchedFrozenDINOv2(backbone_frozen)
        f_state = torch.load(frozen_ckpt, map_location=device)
        self.model_frozen.load_state_dict(f_state['model_state_dict'])
        self.model_frozen.to(device).eval()
        
        backbone_lora = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github')
        self.model_lora = ControlledLoRADINOv2(backbone_lora)
        l_state = torch.load(lora_ckpt, map_location=device)
        self.model_lora.load_state_dict(l_state['model_state_dict'])
        self.model_lora.to(device).eval()
        
        self.rollout_frozen = ViTAttentionRollout(self.model_frozen, device)
        self.rollout_lora = ViTAttentionRollout(self.model_lora, device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"[✓] Both models initialized and loaded successfully!\n")

    def analyze_image(self, image_path: str, true_class: Optional[str] = None) -> Dict[str, any]:
        """
        Runs complete comparative explainability audit on a single MRI image.
        """
        orig_img = Image.open(image_path).convert("RGB")
        tensor_img = self.transform(orig_img).unsqueeze(0).to(self.device)
        
        # Prepare 224x224 original image array for overlays
        orig_resized = orig_img.resize((224, 224))
        orig_np = np.array(orig_resized)
        
        # Frozen model rollout
        map_f, last_f, logits_f = self.rollout_frozen.compute_rollout(tensor_img)
        probs_f = torch.softmax(logits_f, dim=-1).cpu().numpy()[0]
        pred_idx_f = int(probs_f.argmax())
        
        # LoRA model rollout
        map_l, last_l, logits_l = self.rollout_lora.compute_rollout(tensor_img)
        probs_l = torch.softmax(logits_l, dim=-1).cpu().numpy()[0]
        pred_idx_l = int(probs_l.argmax())
        
        # Attention Differential
        delta_map = map_l - map_f
        delta_norm = (delta_map - delta_map.min()) / (delta_map.max() - delta_map.min() + 1e-8)
        
        # Overlays
        def create_overlay(orig, heat, colormap=cv2.COLORMAP_JET, alpha=0.5):
            heat_u8 = np.uint8(255 * heat)
            heat_color = cv2.applyColorMap(heat_u8, colormap)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(orig, 1.0 - alpha, heat_color, alpha, 0)
            
        overlay_f = create_overlay(orig_np, map_f)
        overlay_l = create_overlay(orig_np, map_l)
        overlay_diff = create_overlay(orig_np, delta_norm, colormap=cv2.COLORMAP_MAGMA, alpha=0.6)
        
        # Metrics
        gini_f = compute_gini_index(map_f)
        gini_l = compute_gini_index(map_l)
        ent_f = compute_spatial_entropy(map_f)
        ent_l = compute_spatial_entropy(map_l)
        
        return {
            "image_path": image_path,
            "true_class": true_class,
            "orig_img": orig_np,
            "frozen": {
                "pred_class": CLASS_NAMES[pred_idx_f],
                "confidence": float(probs_f[pred_idx_f]),
                "probs": {CLASS_NAMES[i]: float(probs_f[i]) for i in range(4)},
                "rollout_map": map_f,
                "overlay": overlay_f,
                "gini": gini_f,
                "entropy": ent_f
            },
            "lora": {
                "pred_class": CLASS_NAMES[pred_idx_l],
                "confidence": float(probs_l[pred_idx_l]),
                "probs": {CLASS_NAMES[i]: float(probs_l[i]) for i in range(4)},
                "rollout_map": map_l,
                "overlay": overlay_l,
                "gini": gini_l,
                "entropy": ent_l
            },
            "differential": {
                "delta_map": delta_map,
                "delta_overlay": overlay_diff,
                "gini_gain": gini_l - gini_f,
                "entropy_reduction": ent_f - ent_l
            }
        }

    def render_comparison_figure(self, result: Dict[str, any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Creates publication-ready 4-panel visual comparison:
        [1] Original MRI Scan
        [2] Frozen DINOv2 Attention Rollout
        [3] Controlled QKV-LoRA Attention Rollout
        [4] Attention Shift Differential (LoRA - Frozen)
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='white')
        
        true_cls = result["true_class"] or "Unknown"
        f_cls = result["frozen"]["pred_class"]
        f_conf = result["frozen"]["confidence"] * 100
        l_cls = result["lora"]["pred_class"]
        l_conf = result["lora"]["confidence"] * 100
        
        fname = os.path.basename(result["image_path"])
        
        # Panel 1: Original
        axes[0].imshow(result["orig_img"])
        axes[0].set_title(f"Original MRI: {fname}\nGround Truth: {CLASS_DISPLAY.get(true_cls, true_cls).upper()}", 
                          fontsize=11, fontweight='bold', pad=10)
        axes[0].axis('off')
        
        # Panel 2: Frozen
        f_tag = "✓" if f_cls == true_cls else "❌"
        axes[1].imshow(result["frozen"]["overlay"])
        axes[1].set_title(f"Arm A: Frozen DINOv2\nPred: {f_cls.capitalize()} ({f_conf:.1f}%) {f_tag}\nGini: {result['frozen']['gini']:.3f}", 
                          fontsize=11, fontweight='bold', pad=10)
        axes[1].axis('off')
        
        # Panel 3: LoRA
        l_tag = "✓" if l_cls == true_cls else "❌"
        axes[2].imshow(result["lora"]["overlay"])
        axes[2].set_title(f"Arm B: Controlled LoRA\nPred: {l_cls.capitalize()} ({l_conf:.1f}%) {l_tag}\nGini: {result['lora']['gini']:.3f}", 
                          fontsize=11, fontweight='bold', pad=10)
        axes[2].axis('off')
        
        # Panel 4: Differential
        axes[3].imshow(result["differential"]["delta_overlay"])
        axes[3].set_title(f"Attention Shift (LoRA - Frozen)\nRescued Pathology Focus\nΔGini: {result['differential']['gini_gain']:+.3f}", 
                          fontsize=11, fontweight='bold', pad=10)
        axes[3].axis('off')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[✓] Saved comparison figure to: {save_path}")
        return fig

    def cleanup(self):
        self.rollout_frozen.remove_hooks()
        self.rollout_lora.remove_hooks()
