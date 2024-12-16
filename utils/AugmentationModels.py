import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from random import sample

# class ConvBNRelu3D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(ConvBNRelu3D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.conv = nn.Conv3d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size= self.kernel_size, stride= self.stride, padding= self.padding)
#         self.bn = nn.BatchNorm3d(num_features= self.out_channels)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
# class ConvBNRelu2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(ConvBNRelu2D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.conv = nn.Conv2d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size= self.kernel_size, stride= self.stride, padding= self.padding)
#         self.bn = nn.BatchNorm2d(num_features= self.out_channels)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
# class Spectral_Attention(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super(Spectral_Attention, self).__init__()
#         self.AvgPool = nn.AdaptiveAvgPool3d(1)
#         self.SharedMLP = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.ReLU(),
#             nn.Linear(hidden_features, out_features),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         batch_size, num_channels, D, H, W = x.size()
#         beforfc = self.AvgPool(x)
#         y = beforfc.view(beforfc.size(0), -1)
#         y = self.SharedMLP(y)
#         out = torch.mul(
#             x, y.view(batch_size, num_channels, 1, 1, 1))
#         return out
#
# class Enc_AAE(nn.Module):
#     def __init__(self, channel, output_dim, windowSize):
#         super(Enc_AAE, self).__init__()
#         self.channel = channel
#         self.output_dim = output_dim
#         self.windowSize = windowSize
#         self.conv1 = ConvBNRelu3D(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         self.shotcut = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(1, 1, 1), padding=(0, 0, 0))
#         self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         self.bn = nn.BatchNorm3d(num_features=64)
#         self.relu = nn.ReLU()
#         self.attention = Spectral_Attention(in_features=64, hidden_features=32, out_features=64)
#         self.conv3 = ConvBNRelu2D(in_channels=64*self.channel, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
#         # self.pool = nn.AdaptiveAvgPool2d((3,3))
#         self.projector = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=False)
#         )
#         self.mu = nn.Linear(64, output_dim)
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.attention(x2)
#         x4 = self.shotcut(x)
#         x5 = x4 + x3
#         x5 = self.bn(x5)
#         x5 = self.relu(x5)
#         x6 = x5.reshape([x5.shape[0], -1, x5.shape[3], x5.shape[4]])
#         x7 = self.conv3(x6)
#         # map = self.pool(x10)
#         map = x7
#         h = map.reshape([map.shape[0], -1])
#         x11 = self.projector(h)
#         mu = self.mu(x11)
#         return h, mu
#
#
# class Dec_AAE(nn.Module):
#     def __init__(self, channel, input_dim, windowSize):
#         super(Dec_AAE, self).__init__()
#         self.channel = channel
#         self.windowSize = windowSize
#         self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=128, out_features=128)
#         self.relu2 = nn.ReLU()
#         self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64*self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
#         self.bn3 = nn.BatchNorm2d(num_features=64*self.channel)
#         self.relu3 = nn.ReLU()
#         self.deconv4 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         self.bn4 = nn.BatchNorm3d(num_features=64)
#         self.relu4 = nn.ReLU()
#         self.deconv5 = nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         self.bn5 = nn.BatchNorm3d(num_features=1)
#
#     def forward(self, x):
#         x1 = self.fc1(x)
#         x1 = self.relu1(x1)
#         x2 = self.fc2(x1)
#         x2 = self.relu2(x2)
#         x3 = x2.view(-1, 128, 1, 1)
#         x4 = self.deconv3(x3)
#         x4 = self.bn3(x4)
#         x4 = self.relu3(x4)
#         x5 = x4.view(-1, 64, self.channel, self.windowSize, self.windowSize)
#         x6 = self.deconv4(x5)
#         x6 = self.bn4(x6)
#         x6 = self.relu4(x6)
#         x7 = self.deconv5(x6)
#         x7 = self.bn5(x7)
#         return x7
#
#
# class Discriminant(nn.Module):
#     def __init__(self, encoded_dim):
#         super(Discriminant, self).__init__()
#         self.fc1 = nn.Linear(encoded_dim, 256)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(256, encoded_dim)
#     def forward(self, x):
#         x = F.dropout(self.fc1(x), p=0.2, training=self.training)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return torch.tanh(x)


# class ConvBNRelu3D(nn.Module):
#     def __init__(self,in_channels, out_channels, kernel_size, padding, stride):
#         super(ConvBNRelu3D,self).__init__()
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.kernel_size=kernel_size
#         self.padding=padding
#         self.stride=stride
#         self.conv=nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
#         self.bn=nn.BatchNorm3d(num_features=self.out_channels)
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x= self.relu(x)
#         return x
# class ConvBNRelu2D(nn.Module):
#     def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0):
#         super(ConvBNRelu2D,self).__init__()
#         self.stride = stride
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.kernel_size=kernel_size
#         self.padding=padding
#         self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
#         self.bn=nn.BatchNorm2d(num_features=self.out_channels)
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x= self.relu(x)
#         return x
# class Enc_VAE(nn.Module):
#     def __init__(self,channel,output_dim,windowSize):
#         # 调用Module的初始化
#         super(Enc_VAE, self).__init__()
#         self.channel=channel
#         self.output_dim=output_dim
#         self.windowSize=windowSize
#         self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(7,3,3),stride=1,padding=(0,1,1))
#         self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(5,3,3),stride=1,padding=(0,1,1))
#         self.conv3 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=(0,1,1))
#         self.conv4 = ConvBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
#         # self.pool=nn.AdaptiveAvgPool2d((3, 3))
#         self.projector = nn.Sequential(
#             nn.Linear(64*3*3, 32*3*3),
#             nn.ReLU(inplace=False))
#         self.mu=nn.Linear(32*3*3,output_dim)
#         self.log_sigma=nn.Linear(32*3*3,output_dim)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
#         x = self.conv4(x)
#         # map = self.pool(x)
#         map = x
#         h = map.reshape([map.shape[0], -1])
#         x = self.projector(h)
#         mu=self.mu(x)
#         log_sigma=self.log_sigma(x)
#         sigma=torch.exp(log_sigma)
#         return h, mu, sigma
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
# class Dec_VAE(nn.Module):#input (-1,128)
#     def __init__(self, channel=30,windowSize=25,input_dim=64):
#         super(Dec_VAE, self).__init__()
#         self.channel = channel
#         self.windowSize=windowSize
#         self.fc1=nn.Linear(in_features=input_dim, out_features=256)
#         self.relu1=nn.ReLU()
#         self.fc2=nn.Linear(in_features=256,out_features=64*(self.windowSize)*(self.windowSize))
#         self.relu2 = nn.ReLU()
#         #reshape to (-1,64,17,17) and then deconv.
#         self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32 * (self.channel - 12), kernel_size=(3, 3), stride=(1, 1),
#                                           padding=1)  # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
#         self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
#         self.relu3 = nn.ReLU()
#         # reshape to (-1,32,18,19,19)
#         self.deconv4= nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3,3, 3), stride=(1,1, 1),
#                                           padding=(0,1,1))
#         self.bn4 = nn.BatchNorm3d(num_features=16)
#         self.relu4= nn.ReLU()
#         #(-1,16,20,21,21)
#         self.deconv5= nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(5,3, 3), stride=(1,1, 1),
#                                           padding=(0,1,1))
#         self.bn5 = nn.BatchNorm3d(num_features=8)
#         self.relu5= nn.ReLU()
#         #[-1, 8, 24, 23, 23]
#         self.deconv6= nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(7,3, 3), stride=(1,1, 1),
#                                           padding=(0,1,1))
#         self.bn6 = nn.BatchNorm3d(num_features=1)
#         self.relu6= nn.ReLU()
#         #[-1,1,30,25,25]
#         self._initialize_weights()
#     def forward(self, x):
#         x=self.fc1(x)
#         x=self.relu1(x)
#         x=self.fc2(x)
#         x=self.relu2(x)
#         x=x.view(-1,64,self.windowSize,self.windowSize)
#         x = self.deconv3(x)
#         x = self.bn3(x)
#         x=self.relu3(x)
#         x=x.view(-1,32,self.channel-12,self.windowSize,self.windowSize)
#         x = self.deconv4(x)
#         x = self.bn4(x)
#         x=self.relu4(x)
#         x = self.deconv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
#         x = self.deconv6(x)
#         x = self.bn6(x)
#         # x = self.relu6(x)
#         return x
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#
# class Enc_AAE(nn.Module):
#     def __init__(self,channel,output_dim,windowSize):
#         # 调用Module的初始化
#         super(Enc_AAE, self).__init__()
#         self.channel=channel
#         self.output_dim=output_dim
#         self.windowSize=windowSize
#         self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(7,3,3),stride=1,padding=(0,1,1))
#         self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(5,3,3),stride=1,padding=(0,1,1))
#         self.conv3 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=(0,1,1))
#         self.conv4 = ConvBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
#         # self.pool=nn.AdaptiveAvgPool2d((3, 3))
#         self.projector = nn.Sequential(
#             nn.Linear(64*3*3, 32*3*3),
#             nn.ReLU(inplace=False))
#         self.mu=nn.Linear(32*3*3,output_dim)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
#         x = self.conv4(x)
#         # map = self.pool(x)
#         map = x
#         h = map.reshape([map.shape[0], -1])
#         x = self.projector(h)
#         mu=self.mu(x)
#         return h, mu
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
# class Dec_AAE(nn.Module):#input (-1,128)
#     def __init__(self, channel=30,windowSize=25,input_dim=64):
#         super(Dec_AAE, self).__init__()
#         self.channel = channel
#         self.windowSize=windowSize
#         self.fc1=nn.Linear(in_features=input_dim, out_features=256)
#         self.relu1=nn.ReLU()
#         self.fc2=nn.Linear(in_features=256,out_features=64*(self.windowSize)*(self.windowSize))
#         self.relu2 = nn.ReLU()
#         #reshape to (-1,64,17,17) and then deconv.
#         self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32 * (self.channel - 12), kernel_size=(3, 3), stride=(1, 1),
#                                           padding=1)  # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
#         self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
#         self.relu3 = nn.ReLU()
#         # reshape to (-1,32,18,19,19)
#         self.deconv4= nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3,3, 3), stride=(1,1, 1),
#                                           padding=(0,1,1))
#         self.bn4 = nn.BatchNorm3d(num_features=16)
#         self.relu4= nn.ReLU()
#         #(-1,16,20,21,21)
#         self.deconv5= nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(5,3, 3), stride=(1,1, 1),
#                                           padding=(0,1,1))
#         self.bn5 = nn.BatchNorm3d(num_features=8)
#         self.relu5= nn.ReLU()
#         #[-1, 8, 24, 23, 23]
#         self.deconv6= nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(7,3, 3), stride=(1,1, 1),
#                                           padding=(0,1,1))
#         self.bn6 = nn.BatchNorm3d(num_features=1)
#         self.relu6= nn.ReLU()
#         #[-1,1,30,25,25]
#         self._initialize_weights()
#     def forward(self, x):
#         x=self.fc1(x)
#         x=self.relu1(x)
#         x=self.fc2(x)
#         x=self.relu2(x)
#         x=x.view(-1,64,self.windowSize,self.windowSize)
#         x = self.deconv3(x)
#         x = self.bn3(x)
#         x=self.relu3(x)
#         x=x.view(-1,32,self.channel-12,self.windowSize,self.windowSize)
#         x = self.deconv4(x)
#         x = self.bn4(x)
#         x=self.relu4(x)
#         x = self.deconv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
#         x = self.deconv6(x)
#         x = self.bn6(x)
#         # x = self.relu6(x)
#         return x
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
# class Discriminant(nn.Module):
#     def __init__(self,encoded_dim):
#         super(Discriminant, self).__init__()
#         self.lin1 = nn.Linear(encoded_dim, 288)
#         self.relu=nn.ReLU(inplace=False)
#         self.lin2 = nn.Linear(288, 144)
#         self.relu2=nn.ReLU(inplace=False)
#         self.lin3 = nn.Linear(64,1)
#
#     def forward(self, x):
#         x = F.dropout(self.lin1(x), p=0.2, training=self.training)
#         x = self.relu(x)
#         x = F.dropout(self.lin2(x), p=0.2, training=self.training)
#         # x = self.relu2(self.lin3(x))
#         return torch.tanh(x)

# 1D CNN

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

class ConvBNRelu1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNRelu1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv1d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size= self.kernel_size, stride= self.stride, padding= self.padding)
        self.bn = nn.BatchNorm1d(num_features= self.out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RSEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RSEBasicBlock, self).__init__()
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

class DRSEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DRSEBasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=self.in_channels)
        self.conv2 = nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.out_channels)
        self.se = Channel_Attention(in_features=self.out_channels)
        self.shortcut = nn.Sequential()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
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


class Enc_AAE(nn.Module):
    def __init__(self,channel,output_dim,windowSize):
        # 调用Module的初始化
        super(Enc_AAE, self).__init__()
        self.channel=channel
        self.output_dim=output_dim
        self.windowSize=windowSize
        self.conv1 = ConvBNRelu1D(in_channels=1,out_channels=32,kernel_size=3, stride=1, padding=1)
        self.block1 = RSEBasicBlock(in_channels=32, out_channels=32)
        self.conv2 = ConvBNRelu1D(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.block2 = RSEBasicBlock(in_channels=64, out_channels=64)

        self.pool=nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=False))
        self.mu=nn.Linear(32,output_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2(x)
        map = self.pool(x)
        # map = x
        h = map.reshape([map.shape[0], -1])
        x = self.projector(h)
        mu=self.mu(x)
        return h, mu
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Dec_AAE(nn.Module):#input (-1,128)
    def __init__(self, channel=30,windowSize=25,input_dim=64):
        super(Dec_AAE, self).__init__()
        self.channel = channel
        self.windowSize=windowSize
        self.fc1=nn.Linear(in_features=input_dim, out_features=64)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(in_features=64,out_features=64*(self.channel))
        self.relu2 = nn.ReLU()
        #reshape to (-1,64,17,17) and then deconv.
        self.dblock3 = DRSEBasicBlock(in_channels=64, out_channels=64)
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=1,
                                          padding=1)  # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.relu3 = nn.ReLU()
        # reshape to (-1,32,18,19,19)
        self.dblock4 = DRSEBasicBlock(in_channels=32, out_channels=32)
        self.deconv4= nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=3, stride=1,
                                          padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=1)

        #[-1,1,30,25,25]
        self._initialize_weights()
    def forward(self, x):
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=x.view(-1,64,self.channel)
        x = self.dblock3(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x=self.relu3(x)
        x = self.dblock4(x)
        x = self.deconv4(x)
        x = self.bn4(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Discriminant(nn.Module):
    def __init__(self,encoded_dim):
        super(Discriminant, self).__init__()
        self.lin1 = nn.Linear(encoded_dim, encoded_dim*2)
        self.relu=nn.ReLU(inplace=False)
        self.lin2 = nn.Linear(encoded_dim*2, encoded_dim)
        self.relu2=nn.ReLU(inplace=False)
        self.lin3 = nn.Linear(64,1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = self.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = self.relu2(self.lin3(x))
        return torch.tanh(x)









