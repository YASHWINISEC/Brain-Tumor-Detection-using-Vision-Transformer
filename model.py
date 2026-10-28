import timm
import torch.nn as nn
import torch

def create_vit_model(num_classes, device=None):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    if device:
        model = model.to(device)
    return model
