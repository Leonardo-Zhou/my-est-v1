import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels: int, use_dropout: bool = False):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """CycleGAN生成器 - 用于从有高光到无高光的转换"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 output_channels: int = 3,
                 num_residual_blocks: int = 9,
                 num_features: int = 64):
        super(Generator, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 初始卷积块
            nn.Conv2d(input_channels, num_features, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
            
            # 下采样
            nn.Conv2d(num_features, num_features*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(num_features*2, num_features*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features*4),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residual_blocks)]
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 上采样
            nn.ConvTranspose2d(num_features*4, num_features*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(num_features*2, num_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(num_features, output_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        # 残差学习
        features = self.residual_blocks(features)
        # 解码
        output = self.decoder(features)
        return output


class Discriminator(nn.Module):
    """PatchGAN判别器"""
    
    def __init__(self, input_channels: int = 3, num_features: int = 64):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, num_features, normalize=False),
            *discriminator_block(num_features, num_features*2),
            *discriminator_block(num_features*2, num_features*4),
            *discriminator_block(num_features*4, num_features*8, stride=1),
            nn.Conv2d(num_features*8, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)


class IntrinsicDecompositionNet(nn.Module):
    """专门用于内在图像分解的网络"""
    
    def __init__(self, input_channels: int = 3):
        super(IntrinsicDecompositionNet, self).__init__()
        
        # 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 内在颜色分支 (I')
        self.intrinsic_decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Sigmoid()  # 输出内在颜色 [0, 1]
        )
        
        # 光照分支 (S)
        self.shading_decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 7, padding=3),
            nn.Sigmoid()  # 输出光照图 [0, 1]
        )
        
        # 反射分支 (R)
        self.reflection_decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Sigmoid()  # 输出反射 [0, 1]
        )
    
    def forward(self, x):
        # 共享特征提取
        features = self.shared_encoder(x)
        
        # 三个分支的输出
        intrinsic = self.intrinsic_decoder(features)
        shading = self.shading_decoder(features)
        reflection = self.reflection_decoder(features)
        
        # 扩展shading到3通道
        shading = shading.repeat(1, 3, 1, 1)
        
        return intrinsic, shading, reflection


class AttentionModule(nn.Module):
    """注意力模块，用于聚焦高光区域"""
    
    def __init__(self, in_channels: int):
        super(AttentionModule, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # 生成注意力图
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        return out


class HighlightRemovalNet(nn.Module):
    """专门用于高光去除的网络，结合注意力机制"""
    
    def __init__(self, input_channels: int = 4):  # 3通道图像 + 1通道mask
        super(HighlightRemovalNet, self).__init__()
        
        # 编码器with注意力
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 7, padding=3),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                AttentionModule(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                AttentionModule(256)
            )
        ])
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(256) for _ in range(6)]
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x_with_mask):
        # x_with_mask已经包含了图像和mask（4通道）
        
        # 编码
        features = x_with_mask
        encoder_outputs = []
        for encoder_layer in self.encoder:
            features = encoder_layer(features)
            encoder_outputs.append(features)
        
        # 瓶颈
        features = self.bottleneck(features)
        
        # 解码
        output = self.decoder(features)
        
        return output