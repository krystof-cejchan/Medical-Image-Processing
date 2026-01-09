import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        
        if input_channels != 3:
            original_conv1 = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels=input_channels, 
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
            # zprůměrování vah
            with torch.no_grad():
                self.resnet.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2