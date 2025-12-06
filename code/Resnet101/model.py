import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet101(num_classes=2, pretrained=False):
    """
    Generate ResNet-101 model for Rock Glacier Classification.
    """
    weights = None if not pretrained else models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if not pretrained:
        initialize_weights(model)
        
    return model

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)