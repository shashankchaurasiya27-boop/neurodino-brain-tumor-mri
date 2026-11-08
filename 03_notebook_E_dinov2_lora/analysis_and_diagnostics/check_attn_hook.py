import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

def get_attention_map(image_path, model):
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0) # (1, 3, 224, 224)
    
    attn_weights = []
    
    def hook_fn(module, input, output):
        # output is from qkv layer: (B, N, 3 * C)
        B, N, C_total = output.shape
        C = C_total // 3
        
        # In DINOv2 MemEffAttention, qkv output is typically reshaped to (B, N, 3, num_heads, head_dim)
        # Wait, the source code of dinov2 block is:
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)
        
        scale = (C // 12) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        attn_weights.append(attn.detach())
        
    handle = model.blocks[-1].attn.qkv.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(x)
        
    handle.remove()
    
    attn = attn_weights[0] # (1, 12, N, N)
    
    # We want the attention from the CLS token to all patch tokens
    # CLS token is at index 0. We average across all 12 heads.
    # Shape: (1, 12, N, N) -> (12, N, N)
    attn = attn[0]
    
    cls_attn = attn[:, 0, 1:].mean(dim=0) # (N-1)
    
    # Reshape to grid
    num_patches = int(np.sqrt(len(cls_attn)))
    cls_attn = cls_attn.reshape(num_patches, num_patches)
    
    # Plot
    plt.imshow(cls_attn.numpy())
    plt.savefig('attn_test.png')
    print("Saved attn_test.png")

if __name__ == '__main__':
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    
    # Pick a random test image
    import os
    img_path = "dataset_archive2/archive/Testing/glioma/Te-gl_0010.jpg"
    get_attention_map(img_path, model)
