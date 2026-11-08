import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
methods = dir(model)
print("Methods related to attention:")
for m in methods:
    if 'attn' in m.lower() or 'attention' in m.lower():
        print("-", m)
print("\nLet's check if get_last_selfattention is there:")
if 'get_last_selfattention' in methods:
    print("YES! get_last_selfattention exists.")
elif 'get_intermediate_layers' in methods:
    print("get_intermediate_layers exists.")
