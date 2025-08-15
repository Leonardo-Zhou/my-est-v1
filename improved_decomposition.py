import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy import ndimage


class ImprovedHighlightDetector:
    """改进的高光检测器，专门针对内窥镜图像"""
    
    def __init__(self):
        self.methods = ['intensity', 'gradient', 'saturation', 'specular']
        
    def detect_intensity_highlights(self, image: np.ndarray) -> np.ndarray:
        """基于强度的高光检测"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float32) / 255.0
        
        # 使用多个阈值
        thresh_high = np.percentile(gray, 90)  # 前10%的亮度
        thresh_very_high = np.percentile(gray, 95)  # 前5%的亮度
        
        # 创建mask
        mask1 = gray > thresh_high
        mask2 = gray > thresh_very_high
        
        # 组合mask，给更亮的区域更高权重
        highlight_mask = mask1.astype(np.float32) * 0.5 + mask2.astype(np.float32) * 0.5
        
        return highlight_mask
    
    def detect_gradient_highlights(self, image: np.ndarray) -> np.ndarray:
        """基于梯度的高光检测 - 高光区域通常梯度较小"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化
        gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        
        # 高光区域梯度小但亮度高
        gray_norm = gray.astype(np.float32) / 255.0
        highlight_mask = (gray_norm > 0.7) & (gradient_magnitude < 0.3)
        
        return highlight_mask.astype(np.float32)
    
    def detect_saturation_highlights(self, image: np.ndarray) -> np.ndarray:
        """基于饱和度的高光检测"""
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        
        # 高光区域：高亮度 + 低饱和度
        highlight_mask = (value > 0.7) & (saturation < 0.2)
        
        # 添加极高亮度区域
        very_bright = value > 0.85
        highlight_mask = highlight_mask | very_bright
        
        return highlight_mask.astype(np.float32)
    
    def detect_specular_highlights(self, image: np.ndarray) -> np.ndarray:
        """检测镜面反射高光"""
        # 使用Lab颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0] / 255.0  # 亮度通道
        
        # 镜面反射通常是非常亮的小区域
        thresh = np.percentile(L, 93)
        specular_mask = L > thresh
        
        # 使用形态学操作清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        specular_mask = cv2.morphologyEx(specular_mask.astype(np.uint8), 
                                        cv2.MORPH_CLOSE, kernel)
        
        return specular_mask.astype(np.float32)
    
    def detect_combined(self, image: np.ndarray) -> np.ndarray:
        """组合多种方法检测高光"""
        # 确保输入是uint8格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 应用所有检测方法
        intensity_mask = self.detect_intensity_highlights(image)
        gradient_mask = self.detect_gradient_highlights(image)
        saturation_mask = self.detect_saturation_highlights(image)
        specular_mask = self.detect_specular_highlights(image)
        
        # 加权组合
        combined_mask = (intensity_mask * 0.3 + 
                        gradient_mask * 0.2 + 
                        saturation_mask * 0.3 + 
                        specular_mask * 0.2)
        
        # 阈值化和平滑
        combined_mask = combined_mask > 0.3
        combined_mask = ndimage.gaussian_filter(combined_mask.astype(np.float32), sigma=1.0)
        
        # 确保mask不是全黑
        if combined_mask.max() == 0:
            # 如果没有检测到高光，至少标记最亮的5%区域
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = np.percentile(gray, 95)
            combined_mask = (gray > thresh).astype(np.float32)
        
        return combined_mask


class PhysicallyBasedDecomposition:
    """基于物理模型的图像分解"""
    
    def __init__(self):
        self.highlight_detector = ImprovedHighlightDetector()
        
    def decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将图像分解为 I = I' * S + R
        
        Args:
            image: 输入图像 (H, W, 3), 值范围[0, 1]
            
        Returns:
            dict: 包含 'intrinsic', 'shading', 'reflection', 'highlight_mask'
        """
        if image.max() > 1:
            image = image / 255.0
        
        # 步骤1: 检测高光区域
        highlight_mask = self.highlight_detector.detect_combined((image * 255).astype(np.uint8))
        
        # 步骤2: 提取反射分量R
        reflection = self.extract_reflection(image, highlight_mask)
        
        # 步骤3: 去除反射后的图像
        image_no_reflection = image - reflection
        image_no_reflection = np.clip(image_no_reflection, 0, 1)
        
        # 步骤4: 分解I'和S
        intrinsic, shading = self.decompose_intrinsic_shading(image_no_reflection, highlight_mask)
        
        # 步骤5: 优化以确保重构
        intrinsic, shading, reflection = self.optimize_decomposition(
            image, intrinsic, shading, reflection, highlight_mask
        )
        
        return {
            'intrinsic': intrinsic,
            'shading': shading,
            'reflection': reflection,
            'highlight_mask': highlight_mask,
            'reconstruction': intrinsic * shading + reflection
        }
    
    def extract_reflection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """提取反射分量"""
        reflection = np.zeros_like(image)
        
        # 扩展mask以获取高光周围区域
        dilated_mask = cv2.dilate(mask, np.ones((5, 5)), iterations=2)
        
        for c in range(3):
            channel = image[:, :, c]
            
            # 在高光区域，反射是图像与局部平均的差
            local_mean = cv2.GaussianBlur(channel, (15, 15), 5)
            
            # 计算反射
            reflection_channel = channel - local_mean
            reflection_channel = reflection_channel * dilated_mask
            
            # 确保反射为正
            reflection_channel = np.maximum(reflection_channel, 0)
            
            reflection[:, :, c] = reflection_channel
        
        # 应用mask并平滑
        reflection = reflection * np.expand_dims(dilated_mask, -1)
        for c in range(3):
            reflection[:, :, c] = cv2.GaussianBlur(reflection[:, :, c], (5, 5), 1)
        
        return np.clip(reflection, 0, 1)
    
    def decompose_intrinsic_shading(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分解内在颜色和光照"""
        # 使用Retinex理论
        # 假设光照是平滑的，内在颜色包含细节
        
        # 计算光照（低频分量）
        shading = np.zeros((image.shape[0], image.shape[1], 1))
        
        # 使用多尺度方法估计光照
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 多尺度高斯滤波
        scales = [15, 30, 60]
        illumination = np.zeros_like(gray)
        
        for scale in scales:
            blurred = cv2.GaussianBlur(gray, (scale*2+1, scale*2+1), scale)
            illumination += blurred / len(scales)
        
        # 归一化光照
        illumination = illumination / (illumination.max() + 1e-8)
        
        # 确保光照在合理范围
        illumination = np.clip(illumination, 0.3, 1.0)
        shading[:, :, 0] = illumination
        
        # 计算内在颜色
        shading_3ch = np.repeat(shading, 3, axis=2)
        intrinsic = image / (shading_3ch + 1e-8)
        
        # 归一化内在颜色
        intrinsic = np.clip(intrinsic, 0, 1)
        
        # 在非高光区域，保持颜色一致性
        non_highlight = 1 - mask
        for c in range(3):
            mean_color = np.mean(intrinsic[:, :, c][non_highlight > 0.5])
            if not np.isnan(mean_color):
                intrinsic[:, :, c] = intrinsic[:, :, c] * 0.7 + mean_color * 0.3
        
        return intrinsic, shading
    
    def optimize_decomposition(self, original: np.ndarray, 
                              intrinsic: np.ndarray,
                              shading: np.ndarray,
                              reflection: np.ndarray,
                              mask: np.ndarray,
                              num_iter: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """优化分解以确保重构质量"""
        
        for i in range(num_iter):
            # 当前重构
            shading_3ch = np.repeat(shading, 3, axis=2)
            reconstruction = intrinsic * shading_3ch + reflection
            
            # 计算误差
            error = original - reconstruction
            
            # 更新反射（主要在高光区域）
            reflection_update = error * np.expand_dims(mask, -1) * 0.5
            reflection = reflection + reflection_update
            reflection = np.clip(reflection, 0, 1)
            
            # 更新光照（在非高光区域）
            non_highlight = 1 - mask
            shading_update = np.mean(error * np.expand_dims(non_highlight, -1), axis=2, keepdims=True)
            shading = shading + shading_update * 0.3
            shading = np.clip(shading, 0.1, 1.0)
            
            # 平滑光照
            shading[:, :, 0] = cv2.GaussianBlur(shading[:, :, 0], (5, 5), 1)
            
            # 更新内在颜色
            shading_3ch = np.repeat(shading, 3, axis=2)
            intrinsic = (original - reflection) / (shading_3ch + 1e-8)
            intrinsic = np.clip(intrinsic, 0, 1)
        
        return intrinsic, shading, reflection


class SimpleDecomposer:
    """简单但有效的分解器，用于测试"""
    
    def __init__(self):
        pass
    
    def decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """简单分解方法"""
        if image.max() > 1:
            image = image / 255.0
        
        # 1. 检测高光（简单阈值）
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        highlight_mask = gray > 0.8  # 简单阈值
        highlight_mask = cv2.GaussianBlur(highlight_mask.astype(np.float32), (5, 5), 1)
        
        # 2. 提取反射（高光区域的额外亮度）
        reflection = np.zeros_like(image)
        for c in range(3):
            reflection[:, :, c] = image[:, :, c] * highlight_mask * 0.5
        
        # 3. 估计光照（平滑的灰度图）
        shading_gray = cv2.GaussianBlur(gray, (31, 31), 10)
        shading_gray = np.clip(shading_gray, 0.3, 1.0)  # 避免太暗
        shading = shading_gray.reshape(image.shape[0], image.shape[1], 1)
        
        # 4. 计算内在颜色
        image_no_reflection = image - reflection
        image_no_reflection = np.clip(image_no_reflection, 0, 1)
        
        shading_3ch = np.repeat(shading, 3, axis=2)
        intrinsic = image_no_reflection / (shading_3ch + 1e-8)
        intrinsic = np.clip(intrinsic, 0, 1)
        
        # 5. 验证重构
        reconstruction = intrinsic * shading_3ch + reflection
        
        return {
            'intrinsic': intrinsic,
            'shading': shading,
            'reflection': reflection,
            'highlight_mask': highlight_mask,
            'reconstruction': reconstruction,
            'original': image
        }