import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_ESC50(nn.Module):
    def __init__(self, num_classes=50):
        super(CNN_ESC50, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 109, kernel_size=2, stride=1),
            nn.BatchNorm2d(109),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(109, 203, kernel_size=2, stride=1),
            nn.BatchNorm2d(203),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(203, 181, kernel_size=3, stride=1),
            nn.BatchNorm2d(181),
            nn.ReLU()
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(181, 210, kernel_size=4, stride=1),
            nn.BatchNorm2d(210),
            nn.ReLU()
        )
        
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(210, 169, kernel_size=4, stride=1),
            nn.BatchNorm2d(169),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17914, 850),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Linear(850, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)  # -> [B, 109, 56, 56]
        x = self.conv_block2(x)  # -> [B, 203, 18, 18]
        x = self.conv_block3(x)  # -> [B, 181, 18, 18]
        x = self.conv_block4(x)  # -> [B, 210, 18, 18]
        x = self.conv_block5(x)  # -> [B, 169, 18, 18]
        x = self.fc1(x)          # -> [B, 850]
        x = self.fc2(x)          # -> [B, 50]
        return x
