# resnet_model.py

import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self, num_angle_classes=5, num_level_classes=6):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,  512)
        self.angle_head = nn.Linear(512, num_angle_classes)
        self.level_head = nn.Linear(512, num_level_classes)

    def forward(self, x):
        x = self.resnet(x)
        angle_logits = self.angle_head(x)
        level_logits = self.level_head(x)
        return angle_logits, level_logits

    def forward_angle(self, x):
        x = self.resnet(x)
        angle_logits = self.angle_head(x)
        return angle_logits

    def forward_level(self, x):
        x = self.resnet(x)
        level_logits = self.level_head(x)
        return level_logits
