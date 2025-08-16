import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from scipy import ndimage
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')


class EndoSRRBasedDecomposition:
    """
    基于EndoSRR论文的改进分解方法
    EndoSRR使用SAM-Adapter检测高光，然后使用LaMa进行修复
    我们将这个思想扩展到完整的I=I'*S+R分解
    """
    
    def __init__(self, use_iterative: bool = True):
        self.use_iterative = use_iterative
        self.max_iterations = 5  # 类似EndoSRR的迭代策略
        
    def detect_specular_reflection(self, image: np.ndarray, iteration: int = 0) -> Tuple[np.ndarray, float]:
        """
        检测镜面反射区域，类似EndoSRR的方法
        返回mask和反射比率
        """
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # 多尺度检测策略
        masks = []
        
        # 1. 强度阈值检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 自适应阈值，随迭代逐渐降低（类似EndoSRR的策略）
        base_percentile = 95 - iteration * 2  # 每次迭代降低阈值
        thresh = np.percentile(gray, max(base_percentile, 85))
        intensity_mask = (gray > thresh).astype(np.float32)
        masks.append(intensity_mask)
        
        # 2. 饱和度检测
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        
        # 高光区域：高亮度低饱和度
        saturation_mask = (value > 0.8) & (saturation < 0.15)
        masks.append(saturation_mask.astype(np.float32))
        
        # 3. 颜色一致性检测（高光通常是白色）
        rgb_std = np.std(image.astype(np.float32), axis=2)
        color_mask = (rgb_std < 10) & (gray > 200)
        masks.append(color_mask.astype(np.float32))
        
        # 组合所有mask
        combined_mask = np.maximum.reduce(masks)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 计算反射比率（类似EndoSRR）
        ratio = np.sum(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])
        
        return combined_mask.astype(np.float32), ratio
    
    def inpaint_reflection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        修复高光区域，类似LaMa的功能
        这里使用简化的修复方法
        """
        # 膨胀mask以确保完全覆盖高光
        kernel = np.ones((9, 9), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        
        # 使用OpenCV的inpaint方法（替代LaMa）
        inpainted = cv2.inpaint(
            (image * 255).astype(np.uint8) if image.max() <= 1 else image,
            dilated_mask,
            inpaintRadius=7,
            flags=cv2.INPAINT_TELEA
        )
        
        return inpainted / 255.0 if image.max() <= 1 else inpainted
    
    def extract_shading_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        使用Retinex理论提取光照分量
        基于多尺度Retinex (MSR)
        """
        # 转换为对数域
        img_log = np.log(image + 1e-5)
        
        # 多尺度高斯滤波
        scales = [15, 80, 250]  # 小、中、大尺度
        shading = np.zeros_like(img_log)
        
        for scale in scales:
            # 高斯滤波估计光照
            gaussian = cv2.GaussianBlur(img_log, (0, 0), scale)
            shading += gaussian / len(scales)
        
        # 转回线性域
        shading = np.exp(shading)
        
        # 归一化
        shading = shading / (shading.max() + 1e-8)
        
        return shading
    
    def decompose_intrinsic_shading_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        高级内在图像分解
        使用优化方法分离内在颜色和光照
        """
        H, W, C = image.shape
        
        # 确保图像数据类型正确
        if image.dtype != np.uint8:
            image_display = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        else:
            image_display = image
            
        # 使用灰度图进行光照估计
        gray = cv2.cvtColor(image_display, cv2.COLOR_RGB2GRAY)
        
        # 1. 初始光照估计（使用引导滤波）
        r = 30  # 滤波半径
        eps = 0.001
        
        # 使用引导滤波 - 需要opencv-contrib-python
        try:
            shading_init = cv2.ximgproc.guidedFilter(gray, gray, r, eps)
        except AttributeError:
            # 如果ximgproc不可用，使用双边滤波作为后备方案
            shading_init = cv2.bilateralFilter(gray, r, sigmaColor=50, sigmaSpace=50)
        
        # 2. 优化光照使其平滑
        shading_smooth = self.optimize_shading_smoothness(shading_init)
        
        # 归一化到合理范围
        shading_smooth = np.clip(shading_smooth, 0.2, 1.0)
        
        # 3. 计算内在颜色
        shading_3ch = np.stack([shading_smooth] * 3, axis=2)
        intrinsic = image / (shading_3ch + 1e-8)
        
        # 归一化内在颜色
        intrinsic = np.clip(intrinsic, 0, 1)
        
        return intrinsic, shading_smooth.reshape(H, W, 1)
    
    def optimize_shading_smoothness(self, shading: np.ndarray, lambda_smooth: float = 2.0) -> np.ndarray:
        """
        优化光照的平滑性
        使用加权最小二乘法
        """
        H, W = shading.shape
        N = H * W
        
        # 确保shading数据类型正确
        if shading.dtype != np.uint8:
            shading_uint8 = (shading * 255).astype(np.uint8) if shading.max() <= 1 else shading.astype(np.uint8)
        else:
            shading_uint8 = shading
            
        # 构建稀疏矩阵用于平滑约束
        # 这里使用简化版本
        shading_smooth = cv2.bilateralFilter(
            shading_uint8,
            d=15,
            sigmaColor=50,
            sigmaSpace=50
        ) / 255.0
        
        return shading_smooth
    
    def iterative_decomposition(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        迭代分解方法，类似EndoSRR的多次迭代策略
        """
        if image.max() > 1:
            image = image / 255.0
        
        H, W, C = image.shape
        
        # 初始化
        current_image = image.copy()
        accumulated_reflection = np.zeros_like(image)
        final_mask = np.zeros((H, W), dtype=np.float32)
        
        # 迭代检测和去除高光
        for iteration in range(self.max_iterations):
            # 1. 检测当前图像中的高光
            mask, ratio = self.detect_specular_reflection(
                (current_image * 255).astype(np.uint8), 
                iteration
            )
            
            print(f"Iteration {iteration}: Reflection ratio = {ratio:.6f}")
            
            # 2. 更新总mask
            final_mask = np.maximum(final_mask, mask)
            
            # 3. 提取并累积反射分量
            reflection_iter = current_image * np.expand_dims(mask, -1)
            accumulated_reflection += reflection_iter
            
            # 4. 终止条件（类似EndoSRR）
            if ratio < 1.5e-4 or iteration >= self.max_iterations - 1:
                print(f"Stopping at iteration {iteration}")
                break
            
            # 5. 修复当前图像
            current_image = self.inpaint_reflection(current_image, mask)
        
        # 最终的无高光图像
        image_no_reflection = current_image
        
        # 分解内在颜色和光照
        intrinsic, shading = self.decompose_intrinsic_shading_advanced(image_no_reflection)
        
        # 确保满足I = I' * S + R
        shading_3ch = np.repeat(shading, 3, axis=2)
        reconstruction = intrinsic * shading_3ch + accumulated_reflection
        
        # 精细调整以最小化重构误差
        error = image - reconstruction
        accumulated_reflection += error * 0.5  # 将误差分配给反射
        accumulated_reflection = np.clip(accumulated_reflection, 0, 1)
        
        return {
            'intrinsic': intrinsic,
            'shading': shading,
            'reflection': accumulated_reflection,
            'highlight_mask': final_mask,
            'reconstruction': intrinsic * shading_3ch + accumulated_reflection,
            'original': image,
            'image_no_reflection': image_no_reflection
        }
    
    def decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        主分解函数
        """
        if self.use_iterative:
            return self.iterative_decomposition(image)
        else:
            # 单次分解
            if image.max() > 1:
                image = image / 255.0
            
            # 检测高光
            mask, ratio = self.detect_specular_reflection((image * 255).astype(np.uint8))
            
            # 提取反射
            reflection = image * np.expand_dims(mask, -1)
            
            # 修复图像
            image_no_reflection = self.inpaint_reflection(image, mask)
            
            # 分解
            intrinsic, shading = self.decompose_intrinsic_shading_advanced(image_no_reflection)
            
            shading_3ch = np.repeat(shading, 3, axis=2)
            
            return {
                'intrinsic': intrinsic,
                'shading': shading,
                'reflection': reflection,
                'highlight_mask': mask,
                'reconstruction': intrinsic * shading_3ch + reflection,
                'original': image,
                'image_no_reflection': image_no_reflection
            }


class ChromaticityBasedDecomposition:
    """
    基于色度的分解方法
    假设：反射是非彩色的（白色），内在颜色包含所有色彩信息
    """
    
    def __init__(self):
        pass
    
    def decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        基于色度的分解
        """
        if image.max() > 1:
            image = image / 255.0
        
        H, W, C = image.shape
        
        # 1. 转换到色度空间
        # 计算亮度
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        luminance = luminance.reshape(H, W, 1)
        
        # 计算色度（移除亮度信息）
        chromaticity = image / (luminance + 1e-8)
        
        # 2. 检测无色区域（可能是高光）
        rgb_std = np.std(image, axis=2)
        achromatic_mask = rgb_std < 0.05  # 无色区域
        high_luminance_mask = luminance[:, :, 0] > 0.8
        
        # 高光mask：无色且高亮
        highlight_mask = achromatic_mask & high_luminance_mask
        highlight_mask = cv2.GaussianBlur(highlight_mask.astype(np.float32), (5, 5), 1.0)
        
        # 3. 分离反射（假设反射是无色的）
        # 在高光区域，反射等于亮度超出的部分
        median_luminance = np.median(luminance[~highlight_mask])
        reflection = np.zeros_like(image)
        
        for c in range(3):
            excess = (luminance[:, :, 0] - median_luminance) * highlight_mask
            excess = np.maximum(excess, 0)
            reflection[:, :, c] = excess
        
        # 4. 去除反射后的图像
        image_no_reflection = image - reflection
        image_no_reflection = np.clip(image_no_reflection, 0, 1)
        
        # 5. 分解光照和内在颜色
        # 光照是去除反射后的亮度
        shading = 0.299 * image_no_reflection[:, :, 0] + \
                 0.587 * image_no_reflection[:, :, 1] + \
                 0.114 * image_no_reflection[:, :, 2]
        
        # 平滑光照
        shading = cv2.GaussianBlur(shading, (31, 31), 10)
        shading = np.clip(shading, 0.1, 1.0).reshape(H, W, 1)
        
        # 内在颜色保持色度信息
        shading_3ch = np.repeat(shading, 3, axis=2)
        intrinsic = image_no_reflection / (shading_3ch + 1e-8)
        intrinsic = np.clip(intrinsic, 0, 1)
        
        return {
            'intrinsic': intrinsic,
            'shading': shading,
            'reflection': reflection,
            'highlight_mask': highlight_mask,
            'reconstruction': intrinsic * shading_3ch + reflection,
            'original': image
        }