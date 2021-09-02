import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)                                           
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)                                           

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu,
                                 self.conv2, self.bn2)

        if in_channels != out_channels or self.stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, self.stride),
                                            nn.BatchNorm2d(out_channels))
        else: 
            self.downsample = None

    def forward(self, x):
        out = self.net(x)
        # Residual connection:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)

class SmallModel(nn.Module):
    def __init__(self, num_classes):
        super(SmallModel, self).__init__()
        
        self.in_channels = 64  
        self.kernel_size = 3
        self.blocks = 4
        self.stride = 2
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        layers = []
        self.base_channels = 64
        for i in range(0, self.blocks):
            out_channels = self.base_channels * (2**i)
            layers.append(ResBlock(self.in_channels, out_channels, 3, stride=self.stride))
            self.in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels, num_classes)     

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv_layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
