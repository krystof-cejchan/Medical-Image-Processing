import torch
import torch.nn as nn
from torchvision import models
import utils.params as params 

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, freeze_base=False):
        super(ResNetClassifier, self).__init__()
        

        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        if input_channels != 3:
            original_conv1 = self.model.conv1
            
            self.model.conv1 = nn.Conv2d(
                in_channels=input_channels, 
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
                     
            with torch.no_grad():
                self.model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


