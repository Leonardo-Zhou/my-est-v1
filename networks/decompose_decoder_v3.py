import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class DecomposeDecoderV3(nn.Module):
    """
    Decomposition decoder that outputs two components:
    - Reflectance (R): material reflectance properties
    - Illumination (L): lighting effects
    
    Following the model: I = R × L
    
    Based on the mirror-link architecture from the paper:
    "Learning Non-Lambertian Object Intrinsics across ShapeNet Categories"
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_R_channels=3, num_output_L_channels=1, use_skips=True):
        super(DecomposeDecoderV3, self).__init__()
        
        self.num_output_R_channels = num_output_R_channels
        self.num_output_L_channels = num_output_L_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        self.num_ch_enc = num_ch_enc
        # Match the decoder channels with the encoder channels from ResNet
        self.num_ch_dec = np.array([64, 64, 128, 256, 512])
        
        # Decoder
        self.convs = OrderedDict()
        
        # Reflectance branch (R)
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Illumination branch (L)
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_L", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # Also add reflectance features for cross-component interaction at finest scale
            if i == 0:
                num_ch_in += self.num_ch_dec[0]  # Add reflectance features
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_L", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Output convolutions
        self.convs[("decompose_R_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_R_channels)
        self.convs[("decompose_L_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_L_channels)
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        self.outputs = {}
        
        # Decoder for Reflectance (R)
        x_R = input_features[-1]  # Start from deepest features
        for i in range(4, -1, -1):
            x_R = self.convs[("upconv_R", i, 0)](x_R)
            x_R = [upsample(x_R)]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                x_R += [input_features[i - 1]]
            x_R = torch.cat(x_R, 1)
            x_R = self.convs[("upconv_R", i, 1)](x_R)
        
        self.outputs[("decompose_R")] = self.sigmoid(self.convs[("decompose_R_conv", 0)](x_R))
        
        # Decoder for Illumination (L)
        x_L = input_features[-1]  # Start from deepest features
        for i in range(4, -1, -1):
            x_L = self.convs[("upconv_L", i, 0)](x_L)
            x_L = [upsample(x_L)]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                x_L += [input_features[i - 1]]
            # Add reflectance features at finest scale for interaction
            if i == 0:
                x_L += [x_R]
            x_L = torch.cat(x_L, 1)
            x_L = self.convs[("upconv_L", i, 1)](x_L)
        
        self.outputs[("decompose_L")] = self.sigmoid(self.convs[("decompose_L_conv", 0)](x_L))
        
        return self.outputs[("decompose_R")], self.outputs[("decompose_L")]