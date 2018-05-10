import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seg = GCN(14, 512)
        self.features = nn.Sequential(
            #512x512x15
            nn.Conv2d(15, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #256x256
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #128x128
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #64x64
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #32x32
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #16x16
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            #8x8
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 8),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        seg = F.sigmoid(self.seg(torch.cat((x, x, x), dim=1)))
        out = self.features(torch.cat((seg, x), dim=1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, seg

class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(GCN, self).__init__()
        self.input_size = input_size
        resnet = models.resnet152(pretrained=True)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample(gcfm1, fm3.size()[2:], mode='bilinear') + gcfm2)  # 32
        fs2 = self.brm6(F.upsample(fs1, fm2.size()[2:], mode='bilinear') + gcfm3)  # 64
        fs3 = self.brm7(F.upsample(fs2, fm1.size()[2:], mode='bilinear') + gcfm4)  # 128
        fs4 = self.brm8(F.upsample(fs3, fm0.size()[2:], mode='bilinear'))  # 256
        out = self.brm9(F.upsample(fs4, self.input_size, mode='bilinear'))  # 512

        return out
