import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from skimage import morphology


class HighlightDetector:
    """检测内窥镜图像中的高光区域"""
    
    def __init__(self, 
                 hsv_threshold: float = 0.95,
                 ycbcr_threshold: float = 0.9,
                 min_area: int = 50):
        self.hsv_threshold = hsv_threshold
        self.ycbcr_threshold = ycbcr_threshold
        self.min_area = min_area
        
    def detect_hsv(self, image: np.ndarray) -> np.ndarray:
        """在HSV空间检测高光"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 提取V通道（亮度）
        v_channel = hsv[:, :, 2] / 255.0
        
        # 检测高亮度区域
        highlight_mask = v_channel > self.hsv_threshold
        
        # 同时考虑低饱和度（高光通常饱和度低）
        s_channel = hsv[:, :, 1] / 255.0
        low_saturation = s_channel < 0.2
        
        # 组合条件
        highlight_mask = highlight_mask & low_saturation
        
        return highlight_mask.astype(np.float32)
    
    def detect_ycbcr(self, image: np.ndarray) -> np.ndarray:
        """在YCbCr空间检测高光"""
        # 转换到YCbCr空间
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # 提取Y通道（亮度）
        y_channel = ycbcr[:, :, 0] / 255.0
        
        # 检测高亮度区域
        highlight_mask = y_channel > self.ycbcr_threshold
        
        # 考虑色度信息 - 高光区域色度通常接近中性
        cb = ycbcr[:, :, 1].astype(np.float32) - 128
        cr = ycbcr[:, :, 2].astype(np.float32) - 128
        chroma_magnitude = np.sqrt(cb**2 + cr**2)
        
        # 高光区域色度较低
        low_chroma = chroma_magnitude < 20
        
        # 组合条件
        highlight_mask = highlight_mask & low_chroma
        
        return highlight_mask.astype(np.float32)
    
    def detect_combined(self, image: np.ndarray) -> np.ndarray:
        """组合HSV和YCbCr检测结果"""
        # 获取两种检测结果
        hsv_mask = self.detect_hsv(image)
        ycbcr_mask = self.detect_ycbcr(image)
        
        # 组合两种方法（取并集）
        combined_mask = np.maximum(hsv_mask, ycbcr_mask)
        
        # 形态学操作去除噪声
        combined_mask = morphology.opening(combined_mask, morphology.disk(2))
        combined_mask = morphology.closing(combined_mask, morphology.disk(3))
        
        # 去除小区域
        combined_mask = self._remove_small_regions(combined_mask)
        
        # 高斯模糊使边缘更平滑
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 1.0)
        
        return combined_mask
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """去除小的高光区域"""
        # 连通组件分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0.5).astype(np.uint8), connectivity=8
        )
        
        # 创建清理后的mask
        cleaned_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                cleaned_mask[labels == i] = mask[labels == i]
        
        return cleaned_mask
    
    def process(self, image: np.ndarray, scale_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理图像并返回高光mask
        
        Args:
            image: 输入图像 (H, W, 3) RGB格式，值范围[0, 255]
            scale_factor: 缩放因子，用于加速处理
            
        Returns:
            highlight_mask: 高光区域mask (H, W)，值范围[0, 1]
            scaled_image: 缩放后的图像
        """
        # 保存原始尺寸
        original_height, original_width = image.shape[:2]
        
        # 缩放图像以加速处理
        if scale_factor != 1.0:
            new_height = int(original_height * scale_factor)
            new_width = int(original_width * scale_factor)
            scaled_image = cv2.resize(image, (new_width, new_height), 
                                     interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image.copy()
        
        # 检测高光
        highlight_mask = self.detect_combined(scaled_image)
        
        # 恢复到原始尺寸
        if scale_factor != 1.0:
            highlight_mask = cv2.resize(highlight_mask, 
                                       (original_width, original_height),
                                       interpolation=cv2.INTER_LINEAR)
        
        return highlight_mask, scaled_image


class AdaptiveHighlightDetector(HighlightDetector):
    """自适应高光检测器，根据图像统计动态调整阈值"""
    
    def __init__(self, base_threshold: float = 0.9, min_area: int = 50):
        super().__init__(hsv_threshold=base_threshold, 
                        ycbcr_threshold=base_threshold, 
                        min_area=min_area)
        self.base_threshold = base_threshold
        
    def compute_adaptive_threshold(self, channel: np.ndarray, percentile: float = 99) -> float:
        """基于图像统计计算自适应阈值"""
        # 计算亮度分布
        p_high = np.percentile(channel, percentile)
        p_median = np.median(channel)
        
        # 动态调整阈值
        if p_high - p_median > 0.3:  # 高对比度图像
            threshold = max(self.base_threshold, p_high * 0.95)
        else:  # 低对比度图像
            threshold = max(self.base_threshold * 0.9, p_high * 0.9)
            
        return min(threshold, 0.99)  # 确保阈值不超过0.99
    
    def detect_hsv(self, image: np.ndarray) -> np.ndarray:
        """自适应HSV高光检测"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2] / 255.0
        
        # 计算自适应阈值
        adaptive_threshold = self.compute_adaptive_threshold(v_channel)
        
        # 检测高光
        highlight_mask = v_channel > adaptive_threshold
        s_channel = hsv[:, :, 1] / 255.0
        low_saturation = s_channel < 0.25
        
        highlight_mask = highlight_mask & low_saturation
        
        return highlight_mask.astype(np.float32)