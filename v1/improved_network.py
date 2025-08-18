import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedIntrinsicDecompositionNet(nn.Module):
    """
    改进的内在分解网络
    主要改进：
    1. 限制intrinsic亮度
    2. 强制shading平滑
    3. 确保重构质量
    """
    
    def __init__(self, input_channels=3):
        super(ImprovedIntrinsicDecompositionNet, self).__init__()
        
        # 共享编码器（减少参数量）
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 内在颜色分支 - 输出受限的内在颜色
        self.intrinsic_branch = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 5, padding=2),
            nn.Sigmoid()  # 限制在[0,1]
        )
        
        # 光照分支 - 输出平滑的光照
        self.shading_branch = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 使用更大的卷积核促进平滑
            nn.Conv2d(64, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # 大卷积核输出平滑结果
            nn.Conv2d(8, 1, 7, padding=3),
            nn.Sigmoid()  # 限制在[0,1]
        )
        
        # 反射分支 - 稀疏的反射
        self.reflection_branch = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.ReLU()  # 反射是非负的
        )
        
    def forward(self, x):
        # 共享特征
        features = self.encoder(x)
        
        # 三个分支
        intrinsic = self.intrinsic_branch(features)
        shading = self.shading_branch(features)
        reflection = self.reflection_branch(features)
        
        # 限制intrinsic的亮度范围 [0.2, 0.8]
        intrinsic = 0.2 + 0.6 * intrinsic
        
        # 限制shading的范围 [0.3, 1.0]
        shading = 0.3 + 0.7 * shading
        
        # 限制reflection的最大值
        reflection = torch.clamp(reflection, 0, 0.5)
        
        return intrinsic, shading, reflection


class ImprovedDecompositionLoss(nn.Module):
    """
    改进的损失函数
    强化重构约束和物理约束
    """
    
    def __init__(self, config):
        super(ImprovedDecompositionLoss, self).__init__()
        self.lambda_recon = config.get('lambda_recon', 2.0)  # 增加重构权重
        self.lambda_smooth = config.get('lambda_smooth', 0.5)  # 增加平滑权重
        self.lambda_sparse = config.get('lambda_sparse', 0.1)
        self.lambda_intrinsic_reg = config.get('lambda_intrinsic_reg', 0.2)
        self.lambda_shading_reg = config.get('lambda_shading_reg', 0.3)
        
    def reconstruction_loss(self, image, intrinsic, shading, reflection):
        """强化的重构损失"""
        shading_3ch = shading.repeat(1, 3, 1, 1)
        reconstruction = intrinsic * shading_3ch + reflection
        
        # L1 + L2损失组合
        l1_loss = F.l1_loss(reconstruction, image)
        l2_loss = F.mse_loss(reconstruction, image)
        
        return l1_loss + 0.5 * l2_loss
    
    def shading_smoothness_loss(self, shading):
        """增强的光照平滑损失"""
        # 一阶导数
        grad_x = torch.abs(shading[:, :, :, 1:] - shading[:, :, :, :-1])
        grad_y = torch.abs(shading[:, :, 1:, :] - shading[:, :, :-1, :])
        
        # 二阶导数（促进更平滑）
        grad_xx = torch.abs(grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1])
        grad_yy = torch.abs(grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :])
        
        return torch.mean(grad_x) + torch.mean(grad_y) + 0.5 * (torch.mean(grad_xx) + torch.mean(grad_yy))
    
    def intrinsic_regularization(self, intrinsic):
        """内在颜色正则化 - 防止过亮"""
        # 限制平均亮度
        mean_brightness = torch.mean(intrinsic)
        brightness_penalty = torch.relu(mean_brightness - 0.6) * 2.0  # 惩罚超过0.6的亮度
        
        # 促进颜色一致性
        std_penalty = torch.mean(torch.std(intrinsic, dim=(2, 3)))
        
        return brightness_penalty + 0.1 * std_penalty
    
    def shading_regularization(self, shading):
        """光照正则化 - 防止包含细节"""
        # 惩罚高频成分
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(shading.device)
        
        laplacian = F.conv2d(shading, kernel, padding=1)
        high_freq_penalty = torch.mean(torch.abs(laplacian))
        
        return high_freq_penalty
    
    def reflection_sparsity_loss(self, reflection):
        """改进的反射稀疏损失"""
        # L1稀疏 + 集中性约束
        l1_sparse = torch.mean(torch.abs(reflection))
        
        # 鼓励反射集中在小区域
        reflection_energy = torch.sum(reflection ** 2, dim=1, keepdim=True)
        concentration = torch.mean(reflection_energy)
        
        return l1_sparse + 0.1 * concentration
    
    def forward(self, image, intrinsic, shading, reflection):
        """计算总损失"""
        losses = {}
        
        # 重构损失（最重要）
        losses['reconstruction'] = self.reconstruction_loss(image, intrinsic, shading, reflection)
        
        # 光照平滑（重要）
        losses['shading_smooth'] = self.shading_smoothness_loss(shading)
        
        # 光照正则化（防止细节）
        losses['shading_reg'] = self.shading_regularization(shading)
        
        # 内在颜色正则化（防止过亮）
        losses['intrinsic_reg'] = self.intrinsic_regularization(intrinsic)
        
        # 反射稀疏
        losses['reflection_sparse'] = self.reflection_sparsity_loss(reflection)
        
        # 总损失
        total_loss = (self.lambda_recon * losses['reconstruction'] +
                     self.lambda_smooth * losses['shading_smooth'] +
                     self.lambda_shading_reg * losses['shading_reg'] +
                     self.lambda_intrinsic_reg * losses['intrinsic_reg'] +
                     self.lambda_sparse * losses['reflection_sparse'])
        
        losses['total'] = total_loss
        
        return losses


class FastDecompositionNet(nn.Module):
    """
    快速分解网络 - 优化性能
    使用深度可分离卷积减少参数
    """
    
    def __init__(self, input_channels=3):
        super(FastDecompositionNet, self).__init__()
        
        # 使用深度可分离卷积的编码器
        self.encoder = nn.Sequential(
            # 第一层正常卷积
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 深度可分离卷积
            self._depthwise_conv(32, 64, stride=2),
            self._depthwise_conv(64, 128, stride=2),
        )
        
        # 轻量级解码器
        self.intrinsic_decoder = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # 1x1卷积降维
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.shading_decoder = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 5, padding=2),  # 大核促进平滑
            nn.Sigmoid()
        )
        
        self.reflection_decoder = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU()
        )
    
    def _depthwise_conv(self, in_channels, out_channels, stride=1):
        """深度可分离卷积"""
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        intrinsic = self.intrinsic_decoder(features)
        shading = self.shading_decoder(features)
        reflection = self.reflection_decoder(features)
        
        # 约束范围
        intrinsic = 0.2 + 0.6 * intrinsic  # [0.2, 0.8]
        shading = 0.3 + 0.7 * shading      # [0.3, 1.0]
        reflection = torch.clamp(reflection, 0, 0.3)  # [0, 0.3]
        
        return intrinsic, shading, reflection