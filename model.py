import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels, affine=True)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels, affine=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.bn1(x)
        y = F.relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = F.relu(y)
        z = self.conv2(y)
        output = x + z     
        return output        # residual connection

class BridgeBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels, affine=True)
        self.conv1 = nn.Conv2d(num_channels, 2*num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2*num_channels, affine=True)
        self.conv2 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.bn1(x)
        y = F.relu(y)
        y = self.conv1(y)     # downsample: H,W --> H/2,W/2; channels: C --> 2C
        y = self.bn2(y)
        y = F.relu(y)
        z = self.conv2(y)
        return z              # no skip connection

class ResNet(nn.Module):

    def __init__(self, n):    # n = number of basic blocks per stage
        super().__init__()

        # first conv
        self.first_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # stage 1: n basic blocks, 16 channels
        list1 = []
        for i in range(n):
            list1.append(BasicBlock(16))
        self.list_stage1 = nn.ModuleList(list1)

        # bridge 1: 16 --> 32, 32x32 --> 16x16
        self.bridge1 = BridgeBlock(16)

        # stage 2: n basic blocks, 32 channels
        list2 = []
        for i in range(n):
            list2.append(BasicBlock(32))
        self.list_stage2 = nn.ModuleList(list2)

        # bridge 2: 32 --> 64, 16x16 --> 8x8
        self.bridge2 = BridgeBlock(32)

        # stage 3: n basic blocks, 64 channels
        list3 = []
        for i in range(n):
            list3.append(BasicBlock(64))
        self.list_stage3 = nn.ModuleList(list3)

        self.avgpool = nn.AvgPool2d(kernel_size=8)        # average pooling

        self.last_layer = nn.Linear(64, 10, bias=False)   # last linear layer

    def forward(self, x):
        # x: bs x 3 x 32 x 32
        x = self.first_layer(x)              # bs x 16 x 32 x 32

        for block in self.list_stage1:       # stage 1: bs x 16 x 32 x 32
            x = block(x)

        x = self.bridge1(x)                  # bs x 32 x 16 x 16

        for block in self.list_stage2:       # stage 2: bs x 32 x 16 x 16
            x = block(x)

        x = self.bridge2(x)                  # bs x 64 x 8 x 8

        for block in self.list_stage3:       # stage 3: bs x 64 x 8 x 8
            x = block(x)

        x = self.avgpool(x)                  # bs x 64 x 1 x 1
        x = x.view(x.size(0), 64)            # bs x 64

        scores = self.last_layer(x)          # bs x 10
        return scores
