import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
print(model.blocks[-1])
print("\n--- Attention module ---")
print(model.blocks[-1].attn)
