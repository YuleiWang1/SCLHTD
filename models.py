import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.functional import normalize

class Spatial_attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Spatial_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat((avg_out, max_out), 1)
        y = self.conv1(y)
        return self.act(y)

class Channel_Attention(nn.Module):
    def __init__(self, in_features):
        super(Channel_Attention, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool1d(1)
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.Linear(in_features//2, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, D= x.size()
        beforfc = self.AvgPool(x)
        y = beforfc.view(beforfc.size(0), -1)
        y = self.SharedMLP(y)
        out = torch.mul(
            x, y.view(batch_size, num_channels, 1))
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=self.in_channels)
        self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.out_channels)
        self.se = Channel_Attention(in_features=self.out_channels)
        self.shortcut = nn.Sequential()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(num_features=self.out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.se(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConResNet(nn.Module):
    def __init__(self):
        super(ConResNet, self).__init__()
        # self.attention = Spatial_attention(in_channels=2, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(0,0))
        # self.bn1 = nn.BatchNorm2d(num_features=64)
        # self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.block1 = BasicBlock(in_channels=64, out_channels=64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.block2 = BasicBlock(in_channels=128, out_channels=128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.block3 = BasicBlock(in_channels=64, out_channels=64)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm1d(num_features=1)
        
    def forward(self, x4):
        # x1 = x.reshape(-1, 64, 3, 3)
        # x2 = self.attention(x1)
        # x3 = x2 * x1
        # x3 = x1 + x3
        # x4 = self.conv1(x3)
        # x4 = self.bn1(x4)
        # x4 = self.relu1(x4)
        # x4 = x4.reshape(-1, 1, 128)

        # H = x4.size(0)
        # W = x4.size(1)
        # B = x4.size(2)
        # x4 = x4.reshape(H*W,B)

        
        x4 = torch.unsqueeze(x4, 1)
        x5 = self.conv2(x4)
        x5 = F.relu(self.bn2(x5))
        x6 = self.block1(x5)
        x7 = self.conv3(x6)
        x7 = F.relu(self.bn3(x7))
        x8 = self.block2(x7)
        x9 = self.conv4(x8)
        x9 = F.relu(self.bn4(x9))
        x10 = self.block3(x9)
        x11 = self.conv5(x10)
        x11 = self.bn5(x11)
        out = torch.flatten(x11, 1)
        return out

class SelfSupConNet(nn.Module):
    def __init__(self):
        super(SelfSupConNet, self).__init__()
        self.encoder = ConResNet()
        self.instance_projector = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )  # original= nn.Linear(16,16); Sandiego2=Sandiego100=24; Urban1=26; MUUFL=8
        self.cluster_projector = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    # def forward(self, x_i, x_j):
    #         h_i = self.encoder(x_i)
    #         h_j = self.encoder(x_j)
    #
    #         z_i = normalize(self.instance_projector(h_i), dim=1)
    #         z_j = normalize(self.instance_projector(h_j), dim=1)
    #
    #         c_i = self.cluster_projector(h_i)
    #         c_j = self.cluster_projector(h_j)
    #
    #         return z_i, z_j, c_i, c_j


    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        return z_i, z_j

    def forward_cluster(self, x):
        h = self.encoder(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c




        

        




