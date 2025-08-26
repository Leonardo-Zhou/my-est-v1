import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class NonLambertianDecomposeDecoderV4(nn.Module):
    """
    Non-Lambertian decomposition decoder V4 - 基于原版本成功设计的改进版本
    输出三个组件：
    - Albedo (A): 漫反射反照率
    - Shading (S): 漫反射着色
    - Specular (R): 镜面反射
    
    遵循模型: I = A × S + R
    
    关键改进：
    1. Shading从scale 2开始，符合其平滑特性
    2. 使用合适的通道数配置
    3. Shading输出为灰度图（1通道）
    4. 简化的交叉连接设计
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_A_channels=3, num_output_S_channels=1, num_output_R_channels=3, use_skips=True):
        super(NonLambertianDecomposeDecoderV4, self).__init__()
        
        self.num_output_A_channels = num_output_A_channels
        self.num_output_S_channels = num_output_S_channels  # 保持为1（灰度）
        self.num_output_R_channels = num_output_R_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        self.num_ch_enc = num_ch_enc
        # 使用原版本验证过的通道配置
        self.num_ch_dec = np.array([32, 64, 64, 128, 256])
        
        # Decoder
        self.convs = OrderedDict()
        
        # Albedo branch (A) - 需要高分辨率细节来捕获材质纹理
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # 添加编码器的跳跃连接
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Shading branch (S) - 关键：从scale 2开始，因为shading通常更平滑
        for i in range(2, -1, -1):  # 从scale 2开始，而不是4
            # upconv_0
            if i == 2:
                # 对于shading的第一层，使用来自scale 2的特征
                num_ch_in = self.num_ch_enc[2] if len(self.num_ch_enc) > 2 else self.num_ch_dec[2]
            else:
                num_ch_in = self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_S", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # 在最细尺度添加albedo特征以实现组件间交互
            if i == 0:
                num_ch_in += self.num_ch_dec[0]  # 添加albedo特征
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_S", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Specular branch (R) - 需要高分辨率细节来捕获尖锐的高光
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 输出卷积层
        self.convs[("decompose_A_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_A_channels)
        self.convs[("decompose_S_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_S_channels)  # 1通道灰度图
        self.convs[("decompose_R_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_R_channels)
        
        # 可学习的镜面反射缩放因子，防止其过度主导重建
        self.specular_scale = nn.Parameter(torch.tensor(0.1))  # 从小值开始
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        self.outputs = {}
        
        # Albedo解码器 (A) - 从最深层特征开始
        x_A = input_features[-1]
        for i in range(4, -1, -1):
            x_A = self.convs[("upconv_A", i, 0)](x_A)
            x_A = [upsample(x_A)]
            # 添加编码器的跳跃连接
            if self.use_skips and i > 0:
                x_A += [input_features[i - 1]]
            x_A = torch.cat(x_A, 1)
            x_A = self.convs[("upconv_A", i, 1)](x_A)
        
        self.outputs[("decompose_A")] = self.sigmoid(self.convs[("decompose_A_conv", 0)](x_A))
        
        # Shading解码器 (S) - 关键：从中等分辨率开始
        x_S = input_features[2]  # 从scale 2开始，而不是最深层
        for i in range(2, -1, -1):
            x_S = self.convs[("upconv_S", i, 0)](x_S)
            x_S = [upsample(x_S)]
            if self.use_skips and i > 0:
                x_S += [input_features[i - 1]]
            # 在最细尺度添加albedo特征以实现交互
            if i == 0:
                x_S += [x_A]
            x_S = torch.cat(x_S, 1)
            x_S = self.convs[("upconv_S", i, 1)](x_S)
        
        self.outputs[("decompose_S")] = self.sigmoid(self.convs[("decompose_S_conv", 0)](x_S))
        
        # Specular解码器 (R) - 从最深层特征开始
        x_R = input_features[-1]
        for i in range(4, -1, -1):
            x_R = self.convs[("upconv_R", i, 0)](x_R)
            x_R = [upsample(x_R)]
            if self.use_skips and i > 0:
                x_R += [input_features[i - 1]]
            x_R = torch.cat(x_R, 1)
            x_R = self.convs[("upconv_R", i, 1)](x_R)
        
        # 对镜面反射应用缩放因子，防止其主导重建
        raw_specular = self.sigmoid(self.convs[("decompose_R_conv", 0)](x_R))
        self.outputs[("decompose_R")] = raw_specular * torch.clamp(self.specular_scale, min=0.0, max=1.0)
        
        return self.outputs[("decompose_A")], self.outputs[("decompose_S")], self.outputs[("decompose_R")]