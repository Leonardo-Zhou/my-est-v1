import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class NonLambertianDecomposeDecoderV3(nn.Module):
    """
    Non-Lambertian decomposition decoder that outputs three components:
    - Albedo (A): surface reflectance properties
    - Shading (S): diffuse lighting effects
    - Specular (R): mirror-like reflections
    
    Following the model: I = A × S + R
    
    Based on the mirror-link architecture from the paper:
    "Learning Non-Lambertian Object Intrinsics across ShapeNet Categories"
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_A_channels=3, num_output_S_channels=3, num_output_R_channels=3, use_skips=True):
        super(NonLambertianDecomposeDecoderV3, self).__init__()
        
        self.num_output_A_channels = num_output_A_channels
        self.num_output_S_channels = num_output_S_channels
        self.num_output_R_channels = num_output_R_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        self.num_ch_enc = num_ch_enc
        # Match the decoder channels with the encoder channels from ResNet
        self.num_ch_dec = np.array([64, 64, 128, 256, 512])
        
        # Decoder
        self.convs = OrderedDict()
        
        # Albedo branch (A)
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Shading branch (S)
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_S", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # Also add albedo features for cross-component interaction at finest scale
            if i == 0:
                num_ch_in += self.num_ch_dec[0]  # Add albedo features
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_S", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Specular branch (R)
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
            # Also add albedo and shading features for cross-component interaction at finest scale
            if i == 0:
                num_ch_in += self.num_ch_dec[0] + self.num_ch_dec[0]  # Add albedo and shading features
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Output convolutions
        self.convs[("decompose_A_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_A_channels)
        self.convs[("decompose_S_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_S_channels)
        self.convs[("decompose_R_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_R_channels)
        
        # Learnable scale factor for specular component to prevent it from dominating
        self.specular_scale = nn.Parameter(torch.tensor(0.1))
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        self.outputs = {}
        
        # Decoder for Albedo (A)
        x_A = input_features[-1]  # Start from deepest features
        for i in range(4, -1, -1):
            x_A = self.convs[("upconv_A", i, 0)](x_A)
            x_A = [upsample(x_A)]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                x_A += [input_features[i - 1]]
            x_A = torch.cat(x_A, 1)
            x_A = self.convs[("upconv_A", i, 1)](x_A)
        
        self.outputs[("decompose_A")] = self.sigmoid(self.convs[("decompose_A_conv", 0)](x_A))
        
        # Decoder for Shading (S)
        x_S = input_features[-1]  # Start from deepest features
        for i in range(4, -1, -1):
            x_S = self.convs[("upconv_S", i, 0)](x_S)
            x_S = [upsample(x_S)]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                x_S += [input_features[i - 1]]
            # Add albedo features at finest scale for interaction
            if i == 0:
                x_S += [x_A]
            x_S = torch.cat(x_S, 1)
            x_S = self.convs[("upconv_S", i, 1)](x_S)
        
        self.outputs[("decompose_S")] = self.sigmoid(self.convs[("decompose_S_conv", 0)](x_S))
        
        # Decoder for Specular (R)
        x_R = input_features[-1]  # Start from deepest features
        for i in range(4, -1, -1):
            x_R = self.convs[("upconv_R", i, 0)](x_R)
            x_R = [upsample(x_R)]
            # Add skip connections from encoder
            if self.use_skips and i > 0:
                x_R += [input_features[i - 1]]
            # Add albedo and shading features at finest scale for interaction
            if i == 0:
                x_R += [x_A, x_S]
            x_R = torch.cat(x_R, 1)
            x_R = self.convs[("upconv_R", i, 1)](x_R)
        
        # Apply learnable scale factor to prevent specular from dominating
        raw_specular = self.convs[("decompose_R_conv", 0)](x_R)
        self.outputs[("decompose_R")] = self.sigmoid(raw_specular) * torch.clamp(self.specular_scale, 0.0, 1.0)
        
        return self.outputs[("decompose_A")], self.outputs[("decompose_S")], self.outputs[("decompose_R")]