import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class adjust_net_v4(nn.Module):
    """
    V4版本的调整网络，适配不同通道数的输入：
    - Albedo: 3通道 -> 3通道
    - Shading: 1通道 -> 3通道 (扩展为RGB)
    - Specular: 3通道 -> 3通道
    """
    def __init__(self):
        super(adjust_net_v4, self).__init__()
        
        # Albedo adjustment network (3 -> 3)
        self.adjust_A = self._make_adjustment_net(input_channels=3, output_channels=3)
        
        # Shading adjustment network (1 -> 3) - 将灰度shading扩展为RGB
        self.adjust_S = self._make_adjustment_net(input_channels=1, output_channels=3)
        
        # Specular adjustment network (3 -> 3)
        self.adjust_R = self._make_adjustment_net(input_channels=3, output_channels=3)
        
        # Tanh activation for final outputs
        self.tanh = nn.Tanh()
        
        # Sigmoid activation for ensuring positive values
        self.sigmoid = nn.Sigmoid()
    
    def _make_adjustment_net(self, input_channels, output_channels):
        """创建单个组件的调整网络"""
        convs = OrderedDict()
        convs[("conv", 1)] = ConvBlock(input_channels, 32)
        convs[("conv", 2)] = ConvBlock(32, 32)
        convs[("conv", 3)] = ConvBlock(32, 32)
        convs[("conv", 4)] = nn.Conv2d(32, output_channels, kernel_size=1)
        return nn.Sequential(*list(convs.values()))
    
    def forward(self, input_A, input_S, input_R):
        """
        前向传播
        Args:
            input_A: Albedo (B, 3, H, W)
            input_S: Shading (B, 1, H, W) - 灰度图
            input_R: Specular (B, 3, H, W)
        Returns:
            adjust_A: 调整后的Albedo (B, 3, H, W)
            adjust_S: 调整后的Shading (B, 3, H, W) - 扩展为RGB
            adjust_R: 调整后的Specular (B, 3, H, W)
        """
        # 处理Albedo (3通道 -> 3通道)
        adjust_A = self.adjust_A(input_A)
        adjust_A = self.sigmoid(adjust_A)  # 确保albedo为正值
        
        # 处理Shading (1通道 -> 3通道)
        # 首先确保shading是1通道
        if input_S.shape[1] != 1:
            raise ValueError(f"Expected shading to have 1 channel, but got {input_S.shape[1]} channels")
        
        adjust_S = self.adjust_S(input_S)
        adjust_S = self.sigmoid(adjust_S)  # 确保shading为正值
        
        # 处理Specular (3通道 -> 3通道)
        adjust_R = self.adjust_R(input_R)
        adjust_R = self.sigmoid(adjust_R)  # 确保specular为正值
        
        return adjust_A, adjust_S, adjust_R


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 测试V4版本的adjust_net
    model = adjust_net_v4().cuda()
    model.eval()
    
    # 测试输入
    input_A = torch.randn(4, 3, 256, 320).cuda()  # Albedo: 3通道
    input_S = torch.randn(4, 1, 256, 320).cuda()  # Shading: 1通道 (灰度)
    input_R = torch.randn(4, 3, 256, 320).cuda()  # Specular: 3通道
    
    output_A, output_S, output_R = model(input_A, input_S, input_R)
    
    print("V4 adjust_net test:")
    print("Input shapes:")
    print(f"  Albedo: {input_A.shape}")
    print(f"  Shading: {input_S.shape}")
    print(f"  Specular: {input_R.shape}")
    print("Output shapes:")
    print(f"  Albedo: {output_A.shape}")
    print(f"  Shading: {output_S.shape}")
    print(f"  Specular: {output_R.shape}")
    
    # 验证重建
    reconstructed = output_A * output_S + output_R
    print(f"Reconstructed shape: {reconstructed.shape}")