import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class adjust_net_v4_lite(nn.Module):
    """
    V4 Lite版本的调整网络 - 内存优化版本
    更轻量级的设计，减少内存使用
    """
    def __init__(self):
        super(adjust_net_v4_lite, self).__init__()
        
        # 更轻量级的网络设计
        # Albedo adjustment network (3 -> 3) - 简化版本
        self.adjust_A = self._make_lite_adjustment_net(input_channels=3, output_channels=3)
        
        # Shading adjustment network (1 -> 3) - 简化版本
        self.adjust_S = self._make_lite_adjustment_net(input_channels=1, output_channels=3)
        
        # Specular adjustment network (3 -> 3) - 简化版本
        self.adjust_R = self._make_lite_adjustment_net(input_channels=3, output_channels=3)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def _make_lite_adjustment_net(self, input_channels, output_channels):
        """创建轻量级的调整网络"""
        return nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # 减少通道数
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_channels, kernel_size=1)  # 1x1卷积输出
        )
    
    def forward(self, input_A, input_S, input_R):
        """
        前向传播 - 内存优化版本
        """
        # 处理Albedo (3通道 -> 3通道)
        adjust_A = self.adjust_A(input_A)
        adjust_A = self.sigmoid(adjust_A)
        
        # 处理Shading (1通道 -> 3通道)
        if input_S.shape[1] != 1:
            raise ValueError(f"Expected shading to have 1 channel, but got {input_S.shape[1]} channels")
        
        adjust_S = self.adjust_S(input_S)
        adjust_S = self.sigmoid(adjust_S)
        
        # 处理Specular (3通道 -> 3通道)
        adjust_R = self.adjust_R(input_R)
        adjust_R = self.sigmoid(adjust_R)
        
        return adjust_A, adjust_S, adjust_R


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 测试V4 Lite版本的adjust_net
    model = adjust_net_v4_lite().cuda()
    model.eval()
    
    # 测试输入
    input_A = torch.randn(4, 3, 256, 320).cuda()
    input_S = torch.randn(4, 1, 256, 320).cuda()
    input_R = torch.randn(4, 3, 256, 320).cuda()
    
    output_A, output_S, output_R = model(input_A, input_S, input_R)
    
    print("V4 Lite adjust_net test:")
    print("Input shapes:")
    print(f"  Albedo: {input_A.shape}")
    print(f"  Shading: {input_S.shape}")
    print(f"  Specular: {input_R.shape}")
    print("Output shapes:")
    print(f"  Albedo: {output_A.shape}")
    print(f"  Shading: {output_S.shape}")
    print(f"  Specular: {output_R.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")