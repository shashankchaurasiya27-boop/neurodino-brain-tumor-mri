import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(out.shape)
