from __future__ import division, print_function, absolute_import
import math
import torch
from torch import nn
from torch.nn import functional as F

from GHDC import GHDCblock2, GHDCblock3, GHDCblock4
from deeplab import ASPP

from torchinfo import summary
from Res2Net import *
from Gate import Gate, CrossGate

from DWT_IDWT_layer import DWT_2D
from ca import CBAMLayer
# Some Codes for Network Testing
class Wave(nn.Module):
    def __init__(self, in_planes):
        super(Wave, self).__init__()
        self.conv = nn.Conv2d(in_planes*2, in_planes, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.wave = DWT_2D('haar')
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):

        fll, flh, fhl, fhh = self.wave(x)

        flh = abs(flh)
        fhl = abs(fhl)
        fhh = abs(fhh)

        f_att = flh + fhl + fhh
        out = torch.cat((fll, f_att), dim=1)
        f_att = F.interpolate(out, scale_factor=2, mode='nearest')
        f_att = self.conv(f_att)
        f_att = self.bn(f_att)
        f_att = self.relu(f_att)
        return f_att

class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, e=None):
        super(DoubleConv, self).__init__()
        self.Relu = nn.ReLU(inplace=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if e is not None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,  bias=False)
        else:
            self.conv = None

    def forward(self, x):
        out = self.layer(x)
        if self.conv is not None:
            conv_out = self.conv(x)
        else:
            conv_out = x
        return self.Relu(conv_out+out)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')

class RESNet_ASPP(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(RESNet_ASPP, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.aspp1 = ASPP(64,32,[6,12,18])
        self.aspp2 = ASPP(256, 128, [6, 12, 18])
        self.aspp3 = ASPP(512, 256, [6, 12, 18])
        self.aspp4 = ASPP(1024, 512, [6, 12, 18])
        self.cbr1 = nn.Sequential(nn.Conv2d(64,32,1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True))
        self.cbr2 = nn.Sequential(nn.Conv2d(256, 128, 1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True))
        self.cbr3 = nn.Sequential(nn.Conv2d(512, 256, 1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.cbr4 = nn.Sequential(nn.Conv2d(1024,512,1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))
        self.gate1 = Gate(512)
        self.gate2 = Gate(256)
        self.gate3 = Gate(64)




        # decoder
        self.decoder1 = DoubleConv(1024, 512, 512, e=True)
        self.decoder2 = DoubleConv(1024, 512, 256, e=True)
        self.decoder3 = DoubleConv(512, 256, 64, e=True)
        self.decoder4 = DoubleConv(128, 64, 64, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 64, 64, e=True)
        self.out = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        att1 = self.aspp1(x)
        att2 = self.cbr1(x)
        skip1 = torch.cat((att2, att1), dim=1)

        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        att3 = self.aspp2(x1)
        att4 = self.cbr2(x1)
        skip2 = torch.cat((att3, att4), dim=1)

        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        att5 = self.aspp3(x2)
        att6 = self.cbr3(x2)
        skip3 = torch.cat((att5, att6), dim=1)
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        att7 = self.aspp4(x3)
        att8 = self.cbr4(x3)
        skip4 = torch.cat((att7, att8), dim=1)

        # feature_fusion





        # decoder
        # stage1
        de1 = self.decoder1(skip4)
        ga1 = self.gate1(skip3, de1)
        de1 = self.up(de1)
        cat1 = torch.cat((ga1,de1), dim=1)
        # stage2
        de2 = self.decoder2(cat1)
        ga2 = self.gate2(skip2, de2)
        de2 = self.up(de2)
        cat2 = torch.cat((ga2, de2), dim=1)
        # stage3
        de3 = self.decoder3(cat2)
        ga3 = self.gate3(skip1, de3)
        de3 = self.up(de3)
        cat3 = torch.cat((ga3, de3), dim=1)
        # stage4
        de4 = self.decoder4(cat3)
        de4 = self.up(de4)

        out = self.decoder5(de4)
        out = self.out(out)
        return self.softmax(out)



class RES_U2XNet(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(RES_U2XNet, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.g1 = GHDCblock4(64)
        self.g2 = GHDCblock4(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        g2 = self.g2(x1)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)
        return self.softmax(out)

class U2XNet(nn.Module):
    def __init__(self):

        super(U2XNet, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNet_(nn.Module):
    def __init__(self):

        super(U2XNet, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.g1 = GHDCblock4(64)

        self.g2 = GHDCblock4(256)

        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        g2 = self.g2(x1)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class AllWave(nn.Module):
    def __init__(self):

        super(AllWave, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave1 = Wave(64)
        self.wave2 = Wave(256)
        self.wave3 = Wave(512)
        self.wave4 = Wave(1024)
        self.g1 = GHDCblock4(64)
        self.g2 = GHDCblock4(256)
        self.g3 = GHDCblock4(512)
        self.g4 = GHDCblock4(1024)
        self.cbr1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.cbr2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.cbr3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))
        self.cbr4 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave1(x)
        g2 = self.g2(x1)
        wave2 = self.wave2(x1)
        g3 = self.g3(x2)
        wave3 = self.wave3(x2)
        g4 = self.g4(x3)
        wave4 = self.wave4(x3)
        g1 = self.cbr1(g1+wave1)
        g2 = self.cbr2(g2+wave2)
        g3 = self.cbr3(g3+wave3)
        att4 = self.cbr4(g4+wave4)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class single_ASPP(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(single_ASPP, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.aspp1 = ASPP(64,32,[6,12,18])
        self.aspp2 = ASPP(256, 128, [6, 12, 18])
        self.aspp3 = ASPP(512, 256, [6, 12, 18])
        self.aspp4 = ASPP(1024, 512, [6, 12, 18])
        self.cbr1 = nn.Sequential(nn.Conv2d(64,32,1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True))
        self.cbr2 = nn.Sequential(nn.Conv2d(256, 128, 1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True))
        self.cbr3 = nn.Sequential(nn.Conv2d(512, 256, 1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.cbr4 = nn.Sequential(nn.Conv2d(1024,512,1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))
        self.gate1 = Gate(512)
        self.gate2 = Gate(256)
        self.gate3 = Gate(64)




        # decoder
        self.decoder1 = DoubleConv(1024, 512, 512, e=True)
        self.decoder2 = DoubleConv(1024, 512, 256, e=True)
        self.decoder3 = DoubleConv(512, 256, 64, e=True)
        self.decoder4 = DoubleConv(128, 64, 64, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 64, 64, e=True)
        self.out = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        att1 = self.aspp1(x)
        att2 = self.cbr1(x)
        skip1 = torch.cat((att2, att1), dim=1)

        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        att3 = self.aspp2(x1)
        att4 = self.cbr2(x1)
        skip2 = torch.cat((att3, att4), dim=1)

        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        att5 = self.aspp3(x2)
        att6 = self.cbr3(x2)
        skip3 = torch.cat((att5, att6), dim=1)
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        att7 = self.aspp4(x3)
        att8 = self.cbr4(x3)
        skip4 = torch.cat((att7, att8), dim=1)

        # feature_fusion





        # decoder
        # stage1
        de1 = self.decoder1(skip4)
        ga1 = self.gate1(skip3, de1)
        de1 = self.up(de1)
        cat1 = torch.cat((ga1,de1), dim=1)
        # stage2
        de2 = self.decoder2(cat1)
        ga2 = self.gate2(skip2, de2)
        de2 = self.up(de2)
        cat2 = torch.cat((ga2, de2), dim=1)
        # stage3
        de3 = self.decoder3(cat2)
        ga3 = self.gate3(skip1, de3)
        de3 = self.up(de3)
        cat3 = torch.cat((ga3, de3), dim=1)
        # stage4
        de4 = self.decoder4(cat3)
        de4 = self.up(de4)

        out = self.decoder5(de4)
        out = self.out(out)
        return self.softmax(out)


class ALLGA(nn.Module):
    def __init__(self):

        super(ALLGA, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.g1 = GHDCblock4(64)
        self.att1 = CBAMLayer(64)
        self.g2 = GHDCblock4(256)
        self.att2 = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.att3 = CBAMLayer(512)
        self.g4 = GHDCblock4(1024)
        self.att4 = CBAMLayer(1024)
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 1024)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        g1 = self.att1(g1)
        g2 = self.g2(x1)
        g2 = self.att2(g2)
        g3 = self.g3(x2)
        g3 = self.att3(g3)
        g4 = self.g4(x3)
        g4 = self.att4(g4)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, g4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class ALLGHDC(nn.Module):
    def __init__(self):

        super(ALLGHDC, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.g1 = GHDCblock4(64)

        self.g2 = GHDCblock4(256)

        self.g3 = GHDCblock4(512)

        self.g4 = GHDCblock4(1024)

        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 1024)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)

        g2 = self.g2(x1)

        g3 = self.g3(x2)

        g4 = self.g4(x3)


        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, g4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)



class AllASPP(nn.Module):
    def __init__(self):

        super(AllASPP, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.aspp1 = ASPP(64, 64, [6, 12, 18])
        self.aspp2 = ASPP(256, 256, [6, 12, 18])
        self.aspp3 = ASPP(512, 512, [6, 12, 18])
        self.aspp4 = ASPP(1024, 512, [6, 12, 18])

        self.cbr1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.cbr2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.cbr3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))
        self.cbr4 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]

        wave1 = self.aspp1(x)

        wave2 = self.aspp2(x1)

        wave3 = self.aspp3(x2)

        wave4 = self.aspp4(x3)


        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, wave4)
        fl4 = self.up(fl4)
        g3 = torch.cat((wave3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((wave2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((wave1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNetnowave(nn.Module):
    def __init__(self):

        super(U2XNetnowave, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class U2XNetnoGCA(nn.Module):
    def __init__(self):

        super(U2XNetnoGCA, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = x1
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNetnodc(nn.Module):
    def __init__(self):

        super(U2XNetnodc, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = x2
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class U2XNetnoaspp(nn.Module):
    def __init__(self):

        super(U2XNetnoaspp, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.cbr4 = nn.Sequential(nn.Conv2d(1024, 512, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.cbr4(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)



class UXNet(nn.Module):
    def __init__(self):

        super(UXNet, self).__init__()
        res2net = res2net50_26w_4s(pretrained=False)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.cbr4 = nn.Sequential(nn.Conv2d(1024, 512, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))

        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]

        g1 = x
        g2 = x1

        g3 = x2
        att4 = self.cbr4(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)
class U2XNet_rate(nn.Module):
    def __init__(self):

        super(U2XNet_rate, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock2(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock3(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class UNet(nn.Module):
    def __init__(self):

        super(UNet, self).__init__()
        res2net = res2net50_26w_4s(pretrained=False)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3


        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 512, e=True)
        self.decoder2 = DoubleConv(1024, 512, 256, e=True)
        self.decoder3 = DoubleConv(512, 256, 128, e=True)
        self.decoder4 = DoubleConv(192, 128, 64, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]



        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4 = self.up(de4)
        g3 = torch.cat((x2, fl4), dim=1)
        de3 = self.decoder2(g3)

        # stage3
        fl3 = self.up(de3)
        g2 = torch.cat((x1, fl3), dim=1)
        de2 = self.decoder3(g2)

        # stage4
        fl2 = self.up(de2)
        g1 = torch.cat((x, fl2), dim=1)
        de1 = self.decoder4(g1)

        # output
        fl1 = self.up(de1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class U2Net(nn.Module):
    def __init__(self):

        super(U2Net, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)
class U2XNetGCA(nn.Module):
    def __init__(self):

        super(U2XNetGCA, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.cbr = nn.Sequential(nn.Conv2d(1024,512,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)

        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = x2
        att4 = self.cbr(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)
class U2XNetdwt(nn.Module):
    def __init__(self):

        super(U2XNetdwt, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.cbr4 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = x1

        g3 = x2
        att4 = self.cbr4(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)
class U2XNetghdc(nn.Module):
    def __init__(self):

        super(U2XNetghdc, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.g3 = GHDCblock4(512)

        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cbr = nn.Sequential(nn.Conv2d(1024,512,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x

        g2 = x1

        g3 = self.g3(x2)
        att4 = self.cbr(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class U2XNetaspp(nn.Module):
    def __init__(self):

        super(U2XNetaspp, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3


        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x

        g2 = x1
        g3 = x2
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNet12(nn.Module):
    def __init__(self):

        super(U2XNet12, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.cbr1 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = x2
        att4 = self.cbr1(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNet13(nn.Module):
    def __init__(self):

        super(U2XNet13, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))

        self.g3 = GHDCblock4(512)
        self.cbr1 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = x1
        g3 = self.g3(x2)
        att4 = self.cbr1(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNet14(nn.Module):
    def __init__(self):

        super(U2XNet14, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock4(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))

        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = x1
        g3 = x2
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)
class U2XNet23(nn.Module):
    def __init__(self):

        super(U2XNet23, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.cbr = nn.Sequential(nn.Conv2d(1024,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock4(512)

        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.cbr(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNet24(nn.Module):
    def __init__(self):

        super(U2XNet24, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.g2 = GHDCblock4(256)
        self.att = CBAMLayer(256)

        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = x2
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class U2XNet34(nn.Module):
    def __init__(self):

        super(U2XNet34, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3

        self.g3 = GHDCblock4(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = x
        g2 = x1
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)


class U2XNetghdc3(nn.Module):
    def __init__(self):

        super(U2XNetghdc3, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock3(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock3(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock3(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)

class U2XNetghdc2(nn.Module):
    def __init__(self):

        super(U2XNetghdc2, self).__init__()
        res2net = res2net50_26w_4s(pretrained=True)

        self.conv1 = res2net.conv1
        self.bn1 = res2net.bn1
        self.relu = res2net.relu
        self.pool = res2net.maxpool
        self.down = DownSample()
        # encoder
        self.encoder1 = res2net.layer1
        self.encoder2 = res2net.layer2
        self.encoder3 = res2net.layer3
        self.wave = Wave(64)
        self.g1 = GHDCblock2(64)
        self.cbr = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.g2 = GHDCblock2(256)
        self.att = CBAMLayer(256)
        self.g3 = GHDCblock2(512)
        self.aspp = ASPP(1024, 512, [4, 8, 12])
        self.down = DownSample()
        self.center = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.cg4 = CrossGate(1024, 512)
        self.cg3 = CrossGate(512, 1024)
        self.cg2 = CrossGate(256, 512)
        self.cg1 = CrossGate(128, 192)
        # decoder
        self.decoder1 = DoubleConv(1024, 1024, 1024, e=True)
        self.decoder2 = DoubleConv(512, 512, 512, e=True)
        self.decoder3 = DoubleConv(256, 256, 256, e=True)
        self.decoder4 = DoubleConv(128, 128, 128, e=True)
        self.up = UpSample()
        self.decoder5 = DoubleConv(64, 32, 2, e=True)


        self.softmax = nn.Softmax2d()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        # stage 1
        x = self.conv1(x)  # [64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        # stage 2
        x1 = self.pool(x)
        x1 = self.encoder1(x1)  # [256, 64, 64]
        # stage 3
        x2 = self.encoder2(x1)  # [512, 32, 32]
        # stage 4
        x3 = self.encoder3(x2)  # [1024, 16, 16]
        g1 = self.g1(x)
        wave1 = self.wave(x)
        g1 = self.cbr(g1+wave1)
        g2 = self.g2(x1)
        g2 = self.att(g2)
        g3 = self.g3(x2)
        att4 = self.aspp(x3)

        center = self.center(x3)
        # decoder
        # stage1
        de4 = self.decoder1(center)

        fl4, fh4 = self.cg4(de4, att4)
        fl4 = self.up(fl4)
        g3 = torch.cat((g3, self.up(fh4)), dim=1)
        de3 = self.decoder2(fl4)

        # stage3
        fl3, fh3 = self.cg3(de3, g3)
        fl3 = self.up(fl3)
        g2 = torch.cat((g2, self.up(fh3)), dim=1)
        de2 = self.decoder3(fl3)

        # stage4
        fl2, fh2 = self.cg2(de2, g2)
        fl2 = self.up(fl2)
        g1 = torch.cat((g1, self.up(fh2)), dim=1)
        de1 = self.decoder4(fl2)

        # output
        fl1, _ = self.cg1(de1, g1)
        fl1 = self.up(fl1)
        out = self.decoder5(fl1)




        return self.softmax(out)
if __name__ == '__main__':
    a = torch.rand(2,3,224,224).to('cuda')
    net = AllASPP().cuda()
    # b = net(a)
    # print(b.size())
    summary(net, (2, 3, 224, 224), device='cuda')
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))

