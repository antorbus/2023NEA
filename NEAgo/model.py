import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

boardsize = 19

class ResidualBlock(nn.Module):
    def __init__(self, planes=256, kernel_size =3 , stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                        stride=stride, padding='same')#, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                        stride=stride, padding='same')#, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class ValueHead(nn.Module):
    def __init__(self, planes_in = 256 ,planes=1, kernel_size =1 , stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(planes_in, planes, kernel_size=kernel_size,
                        stride=stride, padding='same')#, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.fc1 = nn.Linear(boardsize**2, 256)#, bias=False)
        self.fc2 = nn.Linear(256, 1 )#, bias=False)

    def forward(self, x):
        bs = x.shape[0]
        out = self.conv1(x)
        out= self.bn1(out)
        out = F.relu(out)
        out= out.view(bs, boardsize**2)
        out = self.fc1 (out)
        out = F.relu(out)
        out = self.fc2 (out)
        out = torch.tanh(out)
        return out

class PolicyHead(nn.Module):
    def __init__(self,  planes_in=256, planes=2, kernel_size =1 , stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(planes_in, planes, kernel_size=kernel_size,
                            stride=stride, padding='same')#, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.fc1 = nn.Linear( 2*boardsize**2, boardsize**2 )#, bias=False)

    def forward(self, x):
        bs = x.shape[0]
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out= out.view(bs, 2*boardsize**2)
        out = self.fc1(out)
        return out.view(bs, boardsize,boardsize)

class GoModel(nn.Module):
    def __init__(self, in_planes=3, mid_planes=256, blocks = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3,
                            stride=1, padding='same')#, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.layers = nn.ModuleList()
        for _ in range(blocks):
            self.layers.append(ResidualBlock())
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        for block in self.layers:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
