import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)
print("LoRA Model Created Successfully!")
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(out.shape)
