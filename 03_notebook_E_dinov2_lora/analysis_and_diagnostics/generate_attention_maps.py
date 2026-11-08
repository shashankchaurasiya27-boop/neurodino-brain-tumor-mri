import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from torchvision import transforms
from peft import LoraConfig, get_peft_model

def get_first_image(class_dir):
    for f in sorted(os.listdir(class_dir)):
        if f.endswith('.jpg'):
            return os.path.join(class_dir, f)
    return None

def extract_attention(model, img_tensor):
    attn_weights = []
    
    def hook_fn(module, input, output):
        # In DINOv2 MemEffAttention, qkv output is (B, N, 3*C)
        B, N, C_total = output.shape
        C = C_total // 3
        
        # Reshape to (B, N, 3, num_heads, head_dim) and permute for attention calculation
        qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, 12, N, C/12)
        
        scale = (C // 12) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        attn_weights.append(attn.detach())
        
    handle = model.blocks[-1].attn.qkv.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(img_tensor)
        
    handle.remove()
    
    # attn shape: (1, 12, N, N)
    attn = attn_weights[0][0] # (12, N, N)
    
    # We want CLS token attention (index 0) to all spatial patches (index 1:)
    # Average across all 12 attention heads
    cls_attn = attn[:, 0, 1:].mean(dim=0) # Shape: (256,)
    
    # Reshape into 16x16 grid (since 224 / 14 = 16)
    num_patches = int(np.sqrt(len(cls_attn)))
    cls_attn = cls_attn.reshape(num_patches, num_patches)
    return cls_attn.cpu().numpy()

def main():
    print("--- Phase 1: Environment & Data Preparation ---")
    DATA_DIR = Path("dataset_archive2/archive/Testing")
    EXP_DIR = Path("outputs/EXP_LORA")
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("--- Phase 2: Restoring the LoRA Architecture ---")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.load_state_dict(torch.load(EXP_DIR / "checkpoints/lora_best.pth", map_location='cpu'), strict=False)
    backbone.to(device)
    backbone.eval()
    
    print("--- Phase 3: Generating Heatmaps ---")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("LoRA DINOv2 Attention Maps (What the Model is Looking At)", fontsize=20, fontweight='bold')
    
    for i, cls_name in enumerate(classes):
        img_path = get_first_image(DATA_DIR / cls_name)
        
        # Load Original Image for Plotting
        orig_img = Image.open(img_path).convert('RGB')
        orig_img_resized = orig_img.resize((224, 224))
        orig_np = np.array(orig_img_resized)
        
        # Transform for Network
        tensor_img = transform(orig_img).unsqueeze(0).to(device)
        
        # Extract Attention
        attn_grid = extract_attention(backbone, tensor_img)
        
        # Min-max scale attention for better visualization
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
        
        # Upscale 16x16 to 224x224
        attn_map_resized = cv2.resize(attn_grid, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)
        
        # Plot Top Row (Original)
        axes[0, i].imshow(orig_np)
        axes[0, i].set_title(f"Original: {cls_name.upper()}", fontsize=14)
        axes[0, i].axis('off')
        
        # Plot Bottom Row (Attention Overlay)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("LoRA Attention Map", fontsize=14)
        axes[1, i].axis('off')
        
    plt.tight_layout()
    output_path = EXP_DIR / "figures/attention_maps_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Also save to artifacts for the Walkthrough
    artifact_path = "/Users/shashankchaurasia/.gemini/antigravity-ide/brain/6e2eb918-6ec5-4ba5-8b8e-fbad4ceb8cae/attention_maps_grid.png"
    plt.savefig(artifact_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved Attention Maps to {output_path} and {artifact_path}")

if __name__ == "__main__":
    main()
