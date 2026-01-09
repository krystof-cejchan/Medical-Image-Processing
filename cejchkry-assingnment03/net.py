# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.params import IMG_SIZE


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Vrstva 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2) # -> (16, 256, 256)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (16, 128, 128)
        
        # Vrstva 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) # -> (32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (32, 64, 64)
        
        # Vrstva 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # -> (64, 64, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (64, 32, 32)
        
        # Vrstva 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # -> (128, 32, 32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (128, 16, 16)
        
        # Vrstva 5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=IMG_SIZE, kernel_size=3, stride=1, padding=1) # -> (IMG_SIZE, 16, 16)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (IMG_SIZE, 8, 8)

        
        self.fc1_input_features = IMG_SIZE * 8 * 8
        self.fc1 = nn.Linear(self.fc1_input_features, 512)
        self.fc2 = nn.Linear(512, 128)
        
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x