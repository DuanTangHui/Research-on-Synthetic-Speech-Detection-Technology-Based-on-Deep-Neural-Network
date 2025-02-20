import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

# 1.Res-TSSDNet

# ResNet-style module
class RSM1D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)

        self.nin = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx

class RSM2D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(channels_out)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.bn3 = nn.BatchNorm2d(channels_out)

        self.nin = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx

# Res-TSSDNet
class SSDNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.RSM4 = RSM1D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM4(x)
        # x = F.max_pool1d(x, kernel_size=x.shape[-1])
        x = F.max_pool1d(x, kernel_size=375)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# 2D-Res-TSSDNet
class SSDNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.RSM1 = RSM2D(channels_in=16, channels_out=32)
        self.RSM2 = RSM2D(channels_in=32, channels_out=64)
        self.RSM3 = RSM2D(channels_in=64, channels_out=128)
        self.RSM4 = RSM2D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM3(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM4(x)

        # x = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        x = F.avg_pool2d(x, kernel_size=(27, 25))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x



# 2，Inc-TSSDNet
class DilatedCovModule(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        channels_out = int(channels_out/4)
        self.cv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=1, padding=1)
        self.cv2 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=2, padding=2)
        self.cv4 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=4, padding=4)
        self.cv8 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=8, padding=8)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn4 = nn.BatchNorm1d(channels_out)
        self.bn8 = nn.BatchNorm1d(channels_out)

    def forward(self, xx):
        xx1 = F.relu(self.bn1(self.cv1(xx)))
        xx2 = F.relu(self.bn2(self.cv2(xx)))
        xx4 = F.relu(self.bn4(self.cv4(xx)))
        xx8 = F.relu(self.bn8(self.cv8(xx)))
        yy = torch.cat((xx1, xx2, xx4, xx8), dim=1)
        return yy

# SE
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)   # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):   # x: batch_size*channel*Length
        # print(x.size())
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x * y.expand_as(x)

# SGE
class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1) #batch_size*channel*Length
        self.weight = nn.Parameter(torch.zeros(1, groups, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1))
        self.sig = nn.Sigmoid()
        # self.init_weights()


    def forward(self, x):
        b, c, l = x.shape  #batch_size*channel*Length
        x = x.view(b * self.groups, -1, l)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,l
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,l
        t = xn.view(b * self.groups, -1)  # bs*g,l
        t = t - t.mean(dim=1, keepdim=True)  # bs*g,l
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,l
        t = t.view(b, self.groups, l)  # bs,g,l
        t = t * self.weight + self.bias  # bs,g,l
        t = t.view(b * self.groups, 1, l)  # bs*g,1,l
        x = x * self.sig(t)
        x = x.view(b, c, l)
        return x

    # Inc-TSSDNet

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 原模型
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        self.DCM1 = DilatedCovModule(channels_in=16, channels_out=32)
        self.sge1 = SpatialGroupEnhance(groups=8)
        # self.se1 = SELayer(32,reduction=16)
        self.DCM2 = DilatedCovModule(channels_in=32, channels_out=64)
        self.sge2 = SpatialGroupEnhance(groups=16)
        # self.se2 = SELayer(64,reduction=16)
        self.DCM3 = DilatedCovModule(channels_in=64, channels_out=128)
        self.sge3 = SpatialGroupEnhance(groups=32)
        # self.se3 = SELayer(128,reduction=16)
        self.DCM4 = DilatedCovModule(channels_in=128, channels_out=128)
        # self.sge4 = SpatialGroupEnhance(groups=64)
        # self.se4 = SELayer(128, reduction=16)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x, Freq_aug=True):

        # 原模型
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # x = self.se1(F.max_pool1d(self.DCM1(x), kernel_size=4))
        # x = self.se2(F.max_pool1d(self.DCM2(x), kernel_size=4))
        # x = self.se3(F.max_pool1d(self.DCM3(x), kernel_size=4))
        x = self.sge1(F.max_pool1d(self.DCM1(x), kernel_size=4))
        x = self.sge2(F.max_pool1d(self.DCM2(x), kernel_size=4))
        x = self.sge3(F.max_pool1d(self.DCM3(x), kernel_size=4))

        # x = F.max_pool1d(self.DCM4(x), kernel_size=x.shape[-1])
        x = F.max_pool1d(self.DCM4(x), kernel_size=375)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    Res_TSSDNet = SSDNet1D()
    Res_TSSDNet_2D = SSDNet2D()
    Inc_TSSDNet = DilatedNet()

    num_params_1D = sum(i.numel() for i in Res_TSSDNet.parameters() if i.requires_grad)  # 0.35M
    num_params_2D = sum(i.numel() for i in Res_TSSDNet_2D.parameters() if i.requires_grad)  # 0.97M
    num_params_Inc = sum(i.numel() for i in Inc_TSSDNet.parameters() if i.requires_grad)  # 0.09M
    print('Number of learnable params: 1D_Res {}, 2D {}, 1D_Inc: {}.'.format(num_params_1D, num_params_2D,
                                                                             num_params_Inc))

    x1 = torch.randn(2, 1, 96000)  # batch_size * channel * L
    x2 = torch.randn(2, 1, 432, 400)  # batch_size * channel *
    y1 = Res_TSSDNet(x1)
    y2 = Res_TSSDNet_2D(x2)
    y3 = Inc_TSSDNet(x1)

    print('End of Program.')
