"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import sys
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from layers.channel_attention_layer import SE_Conv_Block
from layers.grid_attention_layer import (GridAttentionBlock2D,
                                         MultiAttentionBlock)
from layers.ddg_attention_layer import Dual_Dilated_Gating
from layers.modules import (UnetDsv3, UnetGridGatingSignal3, UpCat, UpCatconv,
                            conv_block)
from layers.nonlocal_layer import NONLocalBlock2D
from layers.scale_attention_layer import scale_atten_convblock
from layers.modules import (unetConv2, unetUp, unetUp_origin, Attention_block, 
Dblock_more_dilate,Dblock,DecoderBlock,GHDCblock1, GHDCblock2, GHDCblock3, GHDCblock4,
GHDCblock5, denseskip, denseconnect, dilated_dense, transition_block, up_conv)
from networks_other import init_weights
import resplus
import dpnplus
from transmodules import VisionTransformer,DecoderTrans,get_b16_config
# codes for some networks in comparison
#need efficienctnet，se-resnext，se-net，MICCAI,CVPR2021 GNN TRANSFORMER SUPERRESOLUTION
nonlinearity = partial(F.relu,inplace=True)


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        
        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        #Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
    
class DinkNet34(nn.Module):#31M
    def __init__(self, num_classes=2, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)

        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.softmax(out)

class DinkNet50(nn.Module):#217M
    def __init__(self, num_classes=2):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
    
class DinkNet101(nn.Module):#236M
    def __init__(self, num_classes=2):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=2):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class denselinknet(nn.Module):
    def __init__(self, num_classes=2):
        super(denselinknet, self).__init__()

        filters = [64, 128, 256, 512]
        densenet = models.densenet121(pretrained=True)
        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.dense1 = densenet.features.denseblock1
        self.trans1 = densenet.features.transition1
        self.dense2 = densenet.features.denseblock2
        self.trans2 = densenet.features.transition2
        self.dense3 = densenet.features.denseblock3
        self.trans3 = densenet.features.transition3
        #self.dense4 = densenet.features.denseblock4
        #self.trans4 = transition_block(1024)

        #self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])
        self.decoder0 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2 ,padding=1,output_padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        xo = self.firstmaxpool(x)
        e1 = self.dense1(xo)
        e1o = self.trans1(e1)
        e2 = self.dense2(e1o)
        e2o = self.trans2(e2)
        e3 = self.dense3(e2o)
        e3o = self.trans3(e3)


        ee0 = xo
        ee1 = e1o
        ee2 = e2o


        # Decoder

        d3 = self.decoder3(e3o) + ee2
        d2 = self.decoder2(d3) + ee1
        d1 = self.decoder1(d2) + ee0
        d0 = self.decoder0(d1)
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)


        return self.softmax(out)

class denseNNet121(nn.Module):
    def __init__(self, num_classes=1):
        super(denseNNet121, self).__init__()

        filters = [64,128, 256, 512, 1024]
        densenet = models.densenet121(pretrained=True)
        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.dense1 = densenet.features.denseblock1
        self.trans1 = densenet.features.transition1
        self.dense2 = densenet.features.denseblock2
        self.trans2 = densenet.features.transition2
        self.dense3 = densenet.features.denseblock3
        self.trans3 = densenet.features.transition3
        self.dense4 = densenet.features.denseblock4
        self.trans4 = transition_block(1024)

        self.desblock1 = denseconnect(64)
        self.desblock2 = denseconnect(128)
        self.desblock3 = denseconnect(256)
        self.desblock4 = denseconnect(512)

        self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])
        self.decoder0 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        xo = self.firstmaxpool(x)
        e1 = self.dense1(xo)
        e1o = self.trans1(e1)
        e2 = self.dense2(e1o)
        e2o = self.trans2(e2)
        e3 = self.dense3(e2o)
        e3o = self.trans3(e3)
        e4 = self.dense4(e3o)
        e4o = self.trans4(e4)

        ee0 = self.desblock1(xo)
        ee1 = self.desblock2(e1o)
        ee2 = self.desblock3(e2o)
        ee3 = self.desblock4(e3o)

        # Decoder
        d4 = self.decoder4(e4o) + ee3
        d3 = self.decoder3(d4) + ee2
        d2 = self.decoder2(d3) + ee1
        d1 = self.decoder1(d2) + ee0
        d0 = self.decoder0(d1)
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class NNet121(nn.Module):
    def __init__(self, num_classes=1):
        super(NNet121, self).__init__()

        filters = [64,128, 256, 512, 1024]
        densenet = models.densenet121(pretrained=True)
        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.dense1 = densenet.features.denseblock1
        self.trans1 = densenet.features.transition1
        self.dense2 = densenet.features.denseblock2
        self.trans2 = densenet.features.transition2
        self.dense3 = densenet.features.denseblock3
        self.trans3 = densenet.features.transition3
        self.dense4 = densenet.features.denseblock4
        self.trans4 = transition_block(1024)

        self.ghdcblock1 = GHDCblock1(64)
        self.ghdcblock2 = GHDCblock2(128)
        self.ghdcblock3 = GHDCblock3(256)
        self.ghdcblock4 = GHDCblock4(512)

        self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])
        self.decoder0 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        xo = self.firstmaxpool(x)
        e1 = self.dense1(xo)
        e1o = self.trans1(e1)
        e2 = self.dense2(e1o)
        e2o = self.trans2(e2)
        e3 = self.dense3(e2o)
        e3o = self.trans3(e3)
        e4 = self.dense4(e3o)
        e4o = self.trans4(e4)

        ee0=self.ghdcblock1(xo)
        ee1=self.ghdcblock2(e1o)
        ee2=self.ghdcblock3(e2o)
        ee3=self.ghdcblock4(e3o)
        ee4=e4o

        # Decoder
        d4 = self.decoder4(ee4) + ee3
        d3 = self.decoder3(d4) + ee2
        d2 = self.decoder2(d3) + ee1
        d1 = self.decoder1(d2) + ee0
        d0 = self.decoder0(d1)
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)


        return torch.sigmoid(out)

class DANetL(nn.Module):#参数量250M
    def __init__(self,num_classes=1,input_size=224):
        super(DANetL, self).__init__()

        filters = [64, 256, 512, 1024 ,2048]
        resnext50 = models.resnext50_32x4d(pretrained=True)
        self.firstconv = resnext50.conv1 
        self.firstbn = resnext50.bn1
        self.firstrelu = resnext50.relu#下采样out64 112
        self.firstmaxpool = resnext50.maxpool #下采样out64 56
        self.encoder1 = resnext50.layer1      #out256 56
        self.encoder2 = resnext50.layer2#下采样out512 28 den512探索特征，通过se特征感知，最后与上升进行门控，消除噪声，融合特征
        self.encoder3 = resnext50.layer3#下采样out1024 14
        self.encoder4 = resnext50.layer4#下采样out2048 7 

        self.desblock1 = denseskip(filters[0])
        self.desblock2 = denseskip(filters[1])
        self.desblock3 = denseskip(filters[2]) #out 512x2 输出2倍通道
        

        self.gateblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        

        self.cablock1 = SE_Conv_Block(filters[0]*2,filters[0],2,input_size)
        self.cablock2 = SE_Conv_Block(filters[1]*2,filters[1],3,input_size)
        self.cablock3 = SE_Conv_Block(filters[2]*2,filters[2],4,input_size)# out 512



        self.dblock1   = Dblock(filters[3])#空洞跳跃，增加感受野      
        self.dblock2   = Dblock(filters[4])                                                 

        self.decoder4 = DecoderBlock(filters[4], filters[3])#直接上升 out1024 36

        #空洞与直接上升相加通过1024 nonlocal
        self.nonlocalblock= NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        self.decoder3 = DecoderBlock(filters[3], filters[2])#out512

        self.decoder2 = DecoderBlock(filters[2], filters[1])#out256

        self.decoder1 = DecoderBlock(filters[1], filters[0])#out64

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)


    def forward(self,x):
                # Encoder
        # Encoder
        x = self.firstconv(x)#out 64 288
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0=x
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x) #out 256 144
        e2 = self.encoder2(e1)#out 512 72
        e3 = self.encoder3(e2)#out 1024 36
        e4 = self.encoder4(e3)#out 2048 18
        
        # Center

        ed0 = self.desblock1(e0)
        ee0,_ = self.cablock1(ed0)
        ed1 = self.desblock2(e1)
        ee1,_ = self.cablock2(ed1)
        ed2 = self.desblock3(e2)
        ee2,_ = self.cablock3(ed2)
        ee3 = self.dblock1(e3)
        ee4 = self.dblock2(e4)
        

        # Decoder
        d3 = self.decoder4(ee4) + ee3
        a3 = self.nonlocalblock(d3)#out 1024 36


        g2,_ = self.gateblock3(ee2,a3)# 512 1024

        d2 = self.decoder3(a3)+g2#out 512 72

        g1,_ = self.gateblock2(ee1,d2)

        d1 = self.decoder2(d2)+g1#out 256 144

        g0,_ = self.gateblock1(ee0,d1)

        d0 = self.decoder1(d1)+g0 #out 64 288

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class DANetS(nn.Module):#参数量60M
    def __init__(self,num_classes=1,input_size=224):
        super(DANetS, self).__init__()

        filters = [64, 128, 256, 512 ,1024]
        resnext = resplus.ResNeXt_32x4d_down()
        self.conv1 = resnext.conv1
        self.bn1 = resnext.bn1 #out 64 224
        self.encoder1 = resnext.layer1 #out128 112
        self.encoder2 = resnext.layer2 #out256 56
        self.encoder3= resnext.layer3  #out512 28
        self.encoder4 = resnext.layer4 #out 1024 14


        self.desblock1 = denseskip(filters[0])
        self.desblock2 = denseskip(filters[1])
        self.desblock3 = denseskip(filters[2]) #out 512x2 输出2倍通道
        

        self.gateblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        

        self.cablock1 = SE_Conv_Block(filters[0]*2,filters[0],1,input_size)
        self.cablock2 = SE_Conv_Block(filters[1]*2,filters[1],2,input_size)
        self.cablock3 = SE_Conv_Block(filters[2]*2,filters[2],3,input_size)# out 512



        self.dblock1   = Dblock(filters[3])#空洞跳跃，增加感受野      
        self.dblock2   = Dblock(filters[4])                                                 

        self.decoder4 = DecoderBlock(filters[4], filters[3])#直接上升 out512 28

        #空洞与直接上升相加通过1024 nonlocal
        self.nonlocalblock= NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        self.decoder3 = DecoderBlock(filters[3], filters[2])#out256

        self.decoder2 = DecoderBlock(filters[2], filters[1])#out128

        self.decoder1 = DecoderBlock(filters[1], filters[0])#out64



        self.finalconv = nn.Conv2d(64, num_classes, 3, padding=1)


    def forward(self,x):
                # Encoder
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        e0=x #out 64 224
        e1 = self.encoder1(x) #out 256 144
        e2 = self.encoder2(e1)#out 512 72
        e3 = self.encoder3(e2)#out 1024 36
        e4 = self.encoder4(e3)#out 2048 18
        
        # Center
        ed0 = self.desblock1(e0)
        ee0,cp0 = self.cablock1(ed0)
        ed1 = self.desblock2(e1)
        ee1,cp1 = self.cablock2(ed1)
        ed2 = self.desblock3(e2)
        ee2,cp2 = self.cablock3(ed2)
        ee3 = self.dblock1(e3)
        ee4 = self.dblock2(e4)
        

        # Decoder
        d3 = self.decoder4(ee4) + ee3
        a3 = self.nonlocalblock(d3)#out 1024 36


        g2,gp2 = self.gateblock3(ee2,a3)# 512 1024

        d2 = self.decoder3(a3)+g2#out 512 72

        g1,gp1 = self.gateblock2(ee1,d2)

        d1 = self.decoder2(d2)+g1#out 256 144

        g0,gp0 = self.gateblock1(ee0,d1)

        d0 = self.decoder1(d1)+g0 #out 64 288


        out = self.finalconv(d0)

        return torch.sigmoid(out)

class DANetS_Max(nn.Module):#参数量60M
    def __init__(self,num_classes=1,input_size=224):
        super(DANetS_Max, self).__init__()

        filters = [64, 128, 256, 512 ,1024]
        resnext = resplus.ResNeXt_32x4d_max()
        self.conv1 = resnext.conv1
        self.bn1 = resnext.bn1 #out 64 224
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.encoder1 = resnext.layer1 #out128 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = resnext.layer2 #out256 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.encoder3= resnext.layer3  #out512 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.encoder4 = resnext.layer4 #out 1024 14 center
        


        self.desblock1 = denseskip(filters[0])
        self.desblock2 = denseskip(filters[1])
        self.desblock3 = denseskip(filters[2]) #out 512x2 输出2倍通道
        

        self.gateblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        

        self.cablock1 = SE_Conv_Block(filters[0]*2,filters[0],1,input_size)
        self.cablock2 = SE_Conv_Block(filters[1]*2,filters[1],2,input_size)
        self.cablock3 = SE_Conv_Block(filters[2]*2,filters[2],3,input_size)# out 512



        self.dblock1   = Dblock(filters[3])#空洞跳跃，增加感受野      
        self.dblock2   = Dblock(filters[4])                                                 

        self.decoder4 = DecoderBlock(filters[4], filters[3])#直接上升 out512 28

        #空洞与直接上升相加通过1024 nonlocal
        self.nonlocalblock= NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        self.decoder3 = DecoderBlock(filters[3], filters[2])#out256

        self.decoder2 = DecoderBlock(filters[2], filters[1])#out128

        self.decoder1 = DecoderBlock(filters[1], filters[0])#out64



        self.finalconv = nn.Conv2d(64, num_classes, 3, padding=1)


    def forward(self,x):
                # Encoder
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        e0=x #out 64 224
        xm  = self.maxpool0(x)
        e1 = self.encoder1(xm) #out 256 144
        e1m = self.maxpool1(e1)
        e2 = self.encoder2(e1m)#out 512 72
        e2m = self.maxpool2(e2)
        e3 = self.encoder3(e2m)#out 1024 36
        e3m = self.maxpool3(e3)
        e4 = self.encoder4(e3m)#out 2048 18
        
        # Center
        ed0 = self.desblock1(e0)
        ee0,cp0 = self.cablock1(ed0)
        ed1 = self.desblock2(e1)
        ee1,cp1 = self.cablock2(ed1)
        ed2 = self.desblock3(e2)
        ee2,cp2 = self.cablock3(ed2)
        ee3 = self.dblock1(e3)
        ee4 = self.dblock2(e4)
        

        # Decoder
        d3 = self.decoder4(ee4) + ee3
        a3 = self.nonlocalblock(d3)#out 1024 36


        g2,gp2 = self.gateblock3(ee2,a3)# 512 1024

        d2 = self.decoder3(a3)+g2#out 512 72

        g1,gp1 = self.gateblock2(ee1,d2)

        d1 = self.decoder2(d2)+g1#out 256 144

        g0,gp0 = self.gateblock1(ee0,d1)

        d0 = self.decoder1(d1)+g0 #out 64 288


        out = self.finalconv(d0)

        return torch.sigmoid(out)

class DANetuS(nn.Module):#对标CA 参数量4m
    def __init__(self,num_classes=1,input_size=224):
        super(DANetuS, self).__init__()

        filters = [16, 32, 64, 128 ,256]
        resnext = resplus.ResNeXt_8x4d_down()
        self.conv1 = resnext.conv1
        self.bn1 = resnext.bn1 #out 16 224
        self.encoder1 = resnext.layer1 #out32 112
        self.encoder2 = resnext.layer2 #out64 56
        self.encoder3= resnext.layer3  #out128 28
        self.encoder4 = resnext.layer4 #out 256 14


        self.desblock1 = denseskip(filters[0])
        self.desblock2 = denseskip(filters[1])
        self.desblock3 = denseskip(filters[2]) #out 512x2 输出2倍通道
        

        self.gateblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        

        self.cablock1 = SE_Conv_Block(filters[0]*2,filters[0],1,input_size)
        self.cablock2 = SE_Conv_Block(filters[1]*2,filters[1],2,input_size)
        self.cablock3 = SE_Conv_Block(filters[2]*2,filters[2],3,input_size)# out 512



        self.dblock1   = Dblock(filters[3])#空洞跳跃，增加感受野      
        self.dblock2   = Dblock(filters[4])                                                 

        self.decoder4 = DecoderBlock(filters[4], filters[3])#直接上升 out512 28

        #空洞与直接上升相加通过1024 nonlocal
        self.nonlocalblock= NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        self.decoder3 = DecoderBlock(filters[3], filters[2])#out256

        self.decoder2 = DecoderBlock(filters[2], filters[1])#out128

        self.decoder1 = DecoderBlock(filters[1], filters[0])#out64



        self.finalconv = nn.Conv2d(16, num_classes, 3, padding=1)


    def forward(self,x):
                # Encoder
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        e0=x #out 64 224
        e1 = self.encoder1(x) #out 256 144
        e2 = self.encoder2(e1)#out 512 72
        e3 = self.encoder3(e2)#out 1024 36
        e4 = self.encoder4(e3)#out 2048 18
        
        # Center
        ed0 = self.desblock1(e0)
        ee0,cp0 = self.cablock1(ed0)
        ed1 = self.desblock2(e1)
        ee1,cp1 = self.cablock2(ed1)
        ed2 = self.desblock3(e2)
        ee2,cp2 = self.cablock3(ed2)
        ee3 = self.dblock1(e3)
        ee4 = self.dblock2(e4)
        

        # Decoder
        d3 = self.decoder4(ee4) + ee3
        a3 = self.nonlocalblock(d3)#out 1024 36


        g2,gp2 = self.gateblock3(ee2,a3)# 512 1024

        d2 = self.decoder3(a3)+g2#out 512 72

        g1,gp1 = self.gateblock2(ee1,d2)

        d1 = self.decoder2(d2)+g1#out 256 144

        g0,gp0 = self.gateblock1(ee0,d1)

        d0 = self.decoder1(d1)+g0 #out 64 288


        out = self.finalconv(d0)

        return torch.sigmoid(out)

class DANetuS_Max(nn.Module):#对标CA 参数量4m
    def __init__(self,num_classes=1,input_size=224):
        super(DANetuS_Max, self).__init__()

        filters = [16, 32, 64, 128 ,256]
        resnext = resplus.ResNeXt_8x4d_max()
        self.conv1 = resnext.conv1
        self.bn1 = resnext.bn1 #out  224
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.encoder1 = resnext.layer1 #out 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = resnext.layer2 #out 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.encoder3= resnext.layer3  #out 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.encoder4 = resnext.layer4 #out 14 center


        self.desblock1 = denseskip(filters[0])
        self.desblock2 = denseskip(filters[1])
        self.desblock3 = denseskip(filters[2]) #out x2 输出2倍通道
        

        self.gateblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        

        self.cablock1 = SE_Conv_Block(filters[0]*2,filters[0],1,input_size)
        self.cablock2 = SE_Conv_Block(filters[1]*2,filters[1],2,input_size)
        self.cablock3 = SE_Conv_Block(filters[2]*2,filters[2],3,input_size)# out 512



        self.dblock1   = Dblock(filters[3])#空洞跳跃，增加感受野      
        self.dblock2   = Dblock(filters[4])                                                 

        self.decoder4 = DecoderBlock(filters[4], filters[3])#直接上升 out512 28

        #空洞与直接上升相加通过1024 nonlocal
        self.nonlocalblock= NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        self.decoder3 = DecoderBlock(filters[3], filters[2])#out256

        self.decoder2 = DecoderBlock(filters[2], filters[1])#out128

        self.decoder1 = DecoderBlock(filters[1], filters[0])#out64



        self.finalconv = nn.Conv2d(16, num_classes, 3, padding=1)


    def forward(self,x):
                # Encoder
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        e0=x #out 64 224
        xm  = self.maxpool0(x)
        e1 = self.encoder1(xm) #out 256 144
        e1m = self.maxpool1(e1)
        e2 = self.encoder2(e1m)#out 512 72
        e2m = self.maxpool2(e2)
        e3 = self.encoder3(e2m)#out 1024 36
        e3m = self.maxpool3(e3)
        e4 = self.encoder4(e3m)#out 2048 18
        
        # Center
        ed0 = self.desblock1(e0)
        ee0,cp0 = self.cablock1(ed0)
        ed1 = self.desblock2(e1)
        ee1,cp1 = self.cablock2(ed1)
        ed2 = self.desblock3(e2)
        ee2,cp2 = self.cablock3(ed2)
        ee3 = self.dblock1(e3)
        ee4 = self.dblock2(e4)
        

        # Decoder
        d3 = self.decoder4(ee4) + ee3
        a3 = self.nonlocalblock(d3)#out 1024 36


        g2,gp2 = self.gateblock3(ee2,a3)# 512 1024

        d2 = self.decoder3(a3)+g2#out 512 72

        g1,gp1 = self.gateblock2(ee1,d2)

        d1 = self.decoder2(d2)+g1#out 256 144

        g0,gp0 = self.gateblock1(ee0,d1)

        d0 = self.decoder1(d1)+g0 #out 64 288


        out = self.finalconv(d0)

        return torch.sigmoid(out)

class DANetuS_DPN_Max(nn.Module):#对标CA 参数量4m
    def __init__(self,num_classes=1,input_size=224):
        super(DANetuS_DPN_Max, self).__init__()

        filters = [16, 32, 64, 128 ,256]
        dpns = dpnplus.DPNS_max()
        self.input_layer = dpns.input_layer
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.encoder1 = dpns.layer1 #out 112
        self.skip1 = dpns.skip1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = dpns.layer2 #out 56
        self.skip2 = dpns.skip2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.encoder3= dpns.layer3  #out 28
        self.skip3 = dpns.skip3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.encoder4 = dpns.layer4 #out 14 center
        self.skip4 = dpns.skip4

        self.desblock1 = denseskip(filters[0])
        self.desblock2 = denseskip(filters[1])
        self.desblock3 = denseskip(filters[2]) #out x2 输出2倍通道
        

        self.gateblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        self.gateblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1,1))
        

        self.cablock1 = SE_Conv_Block(filters[0]*2,filters[0],1,input_size)
        self.cablock2 = SE_Conv_Block(filters[1]*2,filters[1],2,input_size)
        self.cablock3 = SE_Conv_Block(filters[2]*2,filters[2],3,input_size)# out 512



        self.dblock1   = Dblock(filters[3])#空洞跳跃，增加感受野      
        self.dblock2   = Dblock(filters[4])                                                 

        self.decoder4 = DecoderBlock(filters[4], filters[3])#直接上升 out512 28

        #空洞与直接上升相加通过1024 nonlocal
        self.nonlocalblock= NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        self.decoder3 = DecoderBlock(filters[3], filters[2])#out256

        self.decoder2 = DecoderBlock(filters[2], filters[1])#out128

        self.decoder1 = DecoderBlock(filters[1], filters[0])#out64



        self.finalconv = nn.Conv2d(16, num_classes, 3, padding=1)


    def forward(self,x):
                # Encoder
        # Encoder
        x = self.input_layer(x)
        e0=x #out 64 224
        xm  = self.maxpool0(x)
        e1 = self.encoder1(xm) #out 256 144
        e1m = self.maxpool1(torch.cat(e1, dim=1))
        e2 = self.encoder2(e1m)#out 512 72
        e2m = self.maxpool2(torch.cat(e2, dim=1))
        e3 = self.encoder3(e2m)#out 1024 36
        e3m = self.maxpool3(torch.cat(e3, dim=1))
        e4 = self.encoder4(e3m)#out 2048 18
        
        # Center
        ed0 = self.desblock1(e0)
        ee0,cp0 = self.cablock1(ed0)
        ed1 = self.desblock2(self.skip1(e1))
        ee1,cp1 = self.cablock2(ed1)
        ed2 = self.desblock3(self.skip2(e2))
        ee2,cp2 = self.cablock3(ed2)
        ee3 = self.dblock1(self.skip3(e3))
        ee4 = self.dblock2(self.skip4(e4))
        

        # Decoder
        d3 = self.decoder4(ee4) + ee3
        a3 = self.nonlocalblock(d3)#out 1024 36


        g2,gp2 = self.gateblock3(ee2,a3)# 512 1024

        d2 = self.decoder3(a3)+g2#out 512 72

        g1,gp1 = self.gateblock2(ee1,d2)

        d1 = self.decoder2(d2)+g1#out 256 144

        g0,gp0 = self.gateblock1(ee0,d1)

        d0 = self.decoder1(d1)+g0 #out 64 288


        out = self.finalconv(d0)

        return torch.sigmoid(out)

class UNet(nn.Module):#参数量40M

    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        self.outconv1 = nn.Conv2d(filters[0], n_classes, 3, padding=1)

        self.softmax = nn.Softmax2d()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        #out1 = torch.sigmoid(d1)

        out2 = self.softmax(d1)

        return out2

class UNet_2Plus(nn.Module):#参数量47M

    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_2Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        self.softmax = nn.Softmax2d()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return self.softmax(final)
        else:
            return self.softmax(final_4)

class UNet_3Plus_DeepSup(nn.Module):#参数量27M
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        self.softmax = nn.Softmax2d()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256

        out = (d1 + d2 + d3 + d4 + d5) / 5
        return self.softmax(out)

class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        ## -------------Encoder--------------10 3x3
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        self.softmax = nn.Softmax2d()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return self.softmax(d1)


class CANet(nn.Module):#参数量45M  #covid 256 256 其他224 224 CA skin 256 320
    def __init__(self, args=None, in_ch=3, num_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation'):#ca过于复杂的机制导致早期训练困难，但最后效果是很高的。
        super(CANet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = num_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size=(256,256)
    

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling 10 3x3
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
                                                    inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3],drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor = self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor = self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor = self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, num_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)

        up4,_ = self.up4(g_conv4)
        g_conv3,_ = self.attentionblock3(conv3, up4)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3,_ = self.up3(up3)
        g_conv2,_ = self.attentionblock2(conv2, up3)

        # atten2_map = att2.cpu().detach().numpy().astype(np.float)
        # atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, 224 / atten2_map.shape[2],
        #                                                      300 / atten2_map.shape[3]], order=0)

        up2 = self.up_concat2(g_conv2, up3)
        up2,_ = self.up2(up2)
        g_conv1,_ = self.attentionblock1(conv1, up2)

        # atten1_map = att1.cpu().detach().numpy().astype(np.float)
        # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                      300 / atten1_map.shape[3]], order=0)
        up1 = self.up_concat1(g_conv1, up2)
        up1,_ = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)

        return out


class SANet(nn.Module):  # 跨编解码注意力
    def __init__(self, args=None, in_ch=3, num_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation'):
        super(SANet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = num_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = (224, 224)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling 10 3x3
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
                                                    inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[3], inter_channels=filters[3] // 4)

        # upsampling
        self.up_concat4 = UpCatconv(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCatconv(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCatconv(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCatconv(filters[1], filters[0], self.is_deconv)

        self.final = nn.Sequential(nn.Conv2d(filters[0], num_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)

        g_conv3, _ = self.attentionblock3(conv3, g_conv4)

        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, _ = self.attentionblock2(conv2, up3)

        up2 = self.up_concat2(g_conv2, up3)
        g_conv1, _ = self.attentionblock1(conv1, up2)

        up1 = self.up_concat1(g_conv1, up2)

        out = self.final(up1)

        return out

class BaseUNet(nn.Module):#基本组件
    def __init__(self, args=None, in_ch=3,num_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(BaseUNet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = num_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling 10 3x3
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # upsampling
        self.up_concat4 = UpCatconv(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCatconv(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCatconv(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCatconv(filters[1], filters[0], self.is_deconv)

        self.final = nn.Sequential(nn.Conv2d(filters[0], num_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)

        up3 = self.up_concat3(conv3, up4)

        up2 = self.up_concat2(conv2, up3)

        up1 = self.up_concat1(conv1, up2)

        out = self.final(up1)

        return out


class HD2ANet(nn.Module):  # 参数量45M
    def __init__(self, args=None, in_ch=3, num_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(HD2ANet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = num_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = (224, 224)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling 10 3x3
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.DDG1 = Dual_Dilated_Gating(in_size=filters[0], gate_size=filters[1], inter_size=filters[0])
        self.DDG2 = Dual_Dilated_Gating(in_size=filters[1], gate_size=filters[2], inter_size=filters[1])
        self.DDG3 = Dual_Dilated_Gating(in_size=filters[2], gate_size=filters[3], inter_size=filters[2])
        self.DDG4 = Dual_Dilated_Gating(in_size=filters[3], gate_size=filters[4], inter_size=filters[3])

        # upsampling
        self.up_concat4 = UpCatconv(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCatconv(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCatconv(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCatconv(filters[1], filters[0], self.is_deconv)

        self.final = nn.Sequential(nn.Conv2d(filters[0], num_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        ddg4, _, _ = self.DDG4(conv4, center)

        up4 = self.up_concat4(ddg4, center)  # out 256 28

        ddg3, _, _ = self.DDG3(conv3, up4)
        up3 = self.up_concat3(ddg3, up4)  # out 128 56

        ddg2, _, _ = self.DDG2(conv2, up3)
        up2 = self.up_concat2(ddg2, up3)  # out 64 112

        ddg1, _, _ = self.DDG1(conv1, up2)
        up1 = self.up_concat1(ddg1, up2)  # out 32 112

        out = self.final(up1)

        return out

class TransUNet(nn.Module):#跨编解码注意力
    def __init__(self,config = get_b16_config(), in_ch=3,num_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(TransUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = num_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.img_size = (256,320)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling 10 3x3
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # upsampling
        self.up_concat4 = UpCatconv(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCatconv(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCatconv(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCatconv(filters[1], filters[0], self.is_deconv)

        self.ViT = VisionTransformer(config,self.img_size)

        self.decodertrans = DecoderTrans(config)

        self.final = nn.Sequential(nn.Conv2d(filters[0], num_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        center_vit,_ = self.ViT(center)

        center_vit = self.decodertrans(center_vit)

        up4 = self.up_concat4(conv4, center_vit)

        up3 = self.up_concat3(conv3, up4)

        up2 = self.up_concat2(conv2, up3)

        up1 = self.up_concat1(conv1, up2)

        out = self.final(up1)

        return out

class GHDCNet(nn.Module):
    def __init__(self,num_classes=2,is_deconv=False):
        super(GHDCNet, self).__init__()

        filters = [64, 256, 512, 1024]
        resnext50 = models.resnext50_32x4d(pretrained=True)
        self.firstconv = resnext50.conv1 
        self.firstbn = resnext50.bn1
        self.firstrelu = resnext50.relu#下采样out64 112
        self.firstmaxpool = resnext50.maxpool #下采样out64 56
        self.encoder1 = resnext50.layer1      #out256 56
        self.encoder2 = resnext50.layer2#下采样out512 28 den512探索特征，通过se特征感知，最后与上升进行门控，消除噪声，融合特征
        self.encoder3 = resnext50.layer3#下采样out1024 14
        #self.encoder4 = resnext50.layer4#下采样out2048 7  不需要最大程度的抽象

        self.DD = dilated_dense(filters[3])#out1024 14

        self.DDG1 =  Dual_Dilated_Gating(in_size=filters[0], gate_size=filters[1], inter_size=filters[0])
        self.DDG2 =  Dual_Dilated_Gating(in_size=filters[1], gate_size=filters[2], inter_size=filters[1])
        self.DDG3 =  Dual_Dilated_Gating(in_size=filters[2], gate_size=filters[3], inter_size=filters[2])

        self.up_concat3 = UpCatconv(filters[3], filters[2], is_deconv)
        self.up_concat2 = UpCatconv(filters[2], filters[1], is_deconv)
        self.up_concat1 = UpCatconv(filters[1], filters[0], is_deconv)

        self.final = nn.Sequential( 
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, num_classes, 3, padding=1))

        self.softmax = nn.Softmax2d()

        # Initialise weights
        for m in self.final.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)#out 64 112
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0=x
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x) #out 256 56
        e2 = self.encoder2(e1)#out 512 28
        e3 = self.encoder3(e2)#out 1024 14

        dd = self.DD(e3)#out 1024 14

        e2o,_,_ = self.DDG3(e2,dd)
        print(e2o.size())
        up3 = self.up_concat3(e2o, dd)#out 512 28

        e1o,_,_ = self.DDG2(e1,up3)
        up2 = self.up_concat2(e1o, up3)#out 256 56

        e0o,_,_ = self.DDG1(e0,up2)
        up1 = self.up_concat1(e0o, up2)#out 64 112

        out = self.final(up1)

        return self.softmax(out)

class AttUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=2):
        super(AttUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1)

        self.softmax = nn.Softmax2d()


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.softmax(d1)





if __name__ == "__main__":
    a = torch.rand(2,3,256,320)
    net = TransUNet()
    b = net(a)



