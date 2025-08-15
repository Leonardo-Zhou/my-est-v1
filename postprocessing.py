import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


class PostProcessingOptimizer:
    """后处理优化器，用于精炼分解结果"""
    
    def __init__(self, 
                 smoothness_weight: float = 0.1,
                 sparsity_weight: float = 0.05,
                 max_iter: int = 50):
        """
        Args:
            smoothness_weight: 光照平滑性权重
            sparsity_weight: 反射稀疏性权重
            max_iter: 最大迭代次数
        """
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.max_iter = max_iter
        
    def optimize_shading(self, 
                        image: np.ndarray,
                        intrinsic: np.ndarray,
                        reflection: np.ndarray,
                        initial_shading: Optional[np.ndarray] = None) -> np.ndarray:
        """
        优化光照分量S，使其满足平滑性约束
        
        Args:
            image: 原始图像 I (H, W, C)
            intrinsic: 内在颜色 I' (H, W, C)
            reflection: 反射分量 R (H, W, C)
            initial_shading: 初始光照估计 (H, W, 1)
            
        Returns:
            optimized_shading: 优化后的光照 (H, W, 1)
        """
        H, W, C = image.shape
        
        # 初始化光照
        if initial_shading is None:
            # 从重构约束估计初始光照
            residual = image - reflection
            intrinsic_gray = np.mean(intrinsic, axis=2, keepdims=True)
            initial_shading = np.mean(residual, axis=2, keepdims=True) / (intrinsic_gray + 1e-8)
            initial_shading = np.clip(initial_shading, 0, 1)
        
        # 定义优化目标函数
        def objective(S_flat):
            S = S_flat.reshape(H, W, 1)
            
            # 重构误差
            reconstruction = intrinsic * S + reflection
            recon_error = np.sum((image - reconstruction) ** 2)
            
            # 光照平滑性（梯度惩罚）
            S_dx = np.diff(S[:, :, 0], axis=1)
            S_dy = np.diff(S[:, :, 0], axis=0)
            smoothness = np.sum(S_dx ** 2) + np.sum(S_dy ** 2)
            
            # 总损失
            loss = recon_error + self.smoothness_weight * smoothness
            
            return loss
        
        # 梯度函数
        def gradient(S_flat):
            S = S_flat.reshape(H, W, 1)
            
            # 重构误差的梯度
            reconstruction = intrinsic * S + reflection
            grad_recon = -2 * intrinsic * (image - reconstruction)
            
            # 平滑性梯度（使用拉普拉斯算子）
            laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4
            S_smooth_grad = cv2.filter2D(S[:, :, 0], -1, laplacian)
            S_smooth_grad = S_smooth_grad.reshape(H, W, 1)
            
            # 总梯度
            grad = np.sum(grad_recon, axis=2, keepdims=True) + \
                   self.smoothness_weight * S_smooth_grad
            
            return grad.flatten()
        
        # 优化
        S_init = initial_shading.flatten()
        bounds = [(0, 1)] * len(S_init)  # 约束光照在[0, 1]范围
        
        result = minimize(
            objective,
            S_init,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': self.max_iter}
        )
        
        optimized_shading = result.x.reshape(H, W, 1)
        
        # 额外的平滑处理
        optimized_shading = gaussian_filter(optimized_shading[:, :, 0], sigma=1.0)
        optimized_shading = optimized_shading.reshape(H, W, 1)
        
        return optimized_shading
    
    def refine_decomposition(self,
                            image: np.ndarray,
                            intrinsic: np.ndarray,
                            shading: np.ndarray,
                            reflection: np.ndarray,
                            highlight_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        迭代优化分解结果
        
        Args:
            image: 原始图像 (H, W, C)
            intrinsic: 内在颜色 (H, W, C)
            shading: 光照 (H, W, 1)
            reflection: 反射 (H, W, C)
            highlight_mask: 高光mask (H, W)
            
        Returns:
            refined_intrinsic, refined_shading, refined_reflection
        """
        
        for iteration in range(self.max_iter):
            # 步骤1: 固定I'和R，优化S
            shading = self.optimize_shading(image, intrinsic, reflection, shading)
            
            # 步骤2: 固定S和R，优化I'
            shading_3ch = np.repeat(shading, 3, axis=2)
            intrinsic = (image - reflection) / (shading_3ch + 1e-8)
            intrinsic = np.clip(intrinsic, 0, 1)
            
            # 步骤3: 固定I'和S，优化R
            reconstruction = intrinsic * shading_3ch
            reflection = image - reconstruction
            
            # 约束R主要在高光区域
            mask_3ch = np.expand_dims(highlight_mask, -1)
            reflection = reflection * mask_3ch
            reflection = np.clip(reflection, 0, 1)
            
            # 检查收敛
            total_recon = intrinsic * shading_3ch + reflection
            recon_error = np.mean((image - total_recon) ** 2)
            
            if recon_error < 1e-4:
                break
        
        return intrinsic, shading, reflection


class ShadingRefinement:
    """专门用于光照分量的精细化"""
    
    def __init__(self):
        self.bilateral_filter = self._create_bilateral_filter()
        
    def _create_bilateral_filter(self):
        """创建双边滤波器用于保边平滑"""
        def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return bilateral_filter
    
    def refine_shading(self, shading: np.ndarray, guide_image: np.ndarray) -> np.ndarray:
        """
        使用引导滤波精炼光照
        
        Args:
            shading: 初始光照估计 (H, W, 1)
            guide_image: 引导图像 (H, W, C)
            
        Returns:
            refined_shading: 精炼后的光照
        """
        # 转换为单通道
        if shading.ndim == 3:
            shading = shading[:, :, 0]
        
        # 引导滤波参数
        radius = 8
        eps = 0.01
        
        # 应用引导滤波
        guide_gray = cv2.cvtColor(guide_image, cv2.COLOR_RGB2GRAY) if guide_image.ndim == 3 else guide_image
        refined = cv2.ximgproc.guidedFilter(guide_gray, shading, radius, eps)
        
        # 双边滤波进一步平滑
        refined = self.bilateral_filter(refined, d=9, sigma_color=50, sigma_space=50)
        
        # 确保值在有效范围
        refined = np.clip(refined, 0, 1)
        
        return refined.reshape(shading.shape[0], shading.shape[1], 1)


class IterativeRefinement:
    """迭代精炼框架"""
    
    def __init__(self, num_iterations: int = 3):
        self.num_iterations = num_iterations
        self.shading_refiner = ShadingRefinement()
        self.optimizer = PostProcessingOptimizer()
        
    def refine(self,
               image: np.ndarray,
               initial_intrinsic: np.ndarray,
               initial_shading: np.ndarray,
               initial_reflection: np.ndarray,
               highlight_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        迭代精炼分解结果
        
        Returns:
            dict: 包含'intrinsic', 'shading', 'reflection'的字典
        """
        intrinsic = initial_intrinsic.copy()
        shading = initial_shading.copy()
        reflection = initial_reflection.copy()
        
        for i in range(self.num_iterations):
            # 精炼光照
            shading = self.shading_refiner.refine_shading(shading, image)
            
            # 优化分解
            intrinsic, shading, reflection = self.optimizer.refine_decomposition(
                image, intrinsic, shading, reflection, highlight_mask
            )
            
            # 强制稀疏性约束
            reflection = self._enforce_sparsity(reflection, highlight_mask)
        
        return {
            'intrinsic': intrinsic,
            'shading': shading,
            'reflection': reflection
        }
    
    def _enforce_sparsity(self, reflection: np.ndarray, mask: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """强制反射分量的稀疏性"""
        # 在非高光区域应用更强的稀疏性
        sparse_reflection = reflection.copy()
        non_highlight = 1 - mask
        
        # 软阈值
        sparse_reflection = sparse_reflection * mask[:, :, np.newaxis]
        
        # 在非高光区域，只保留显著的反射
        for c in range(3):
            channel = sparse_reflection[:, :, c]
            channel[non_highlight > 0.5] *= 0.1
            sparse_reflection[:, :, c] = channel
            
        return np.clip(sparse_reflection, 0, 1)


class EnergyMinimization:
    """基于能量最小化的后处理"""
    
    def __init__(self):
        self.lambda_smooth = 0.1
        self.lambda_sparse = 0.05
        self.lambda_recon = 1.0
        
    def minimize_energy(self,
                       image: np.ndarray,
                       intrinsic: np.ndarray,
                       shading: np.ndarray,
                       reflection: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        最小化总能量函数
        E = E_recon + λ_smooth * E_smooth + λ_sparse * E_sparse
        """
        H, W, C = image.shape
        
        # 定义总能量
        def total_energy(params):
            # 解包参数
            n_pixels = H * W
            I_prime = params[:n_pixels * C].reshape(H, W, C)
            S = params[n_pixels * C:n_pixels * (C + 1)].reshape(H, W, 1)
            R = params[n_pixels * (C + 1):].reshape(H, W, C)
            
            # 重构能量
            recon = I_prime * S + R
            E_recon = np.sum((image - recon) ** 2)
            
            # 光照平滑能量
            S_dx = np.diff(S[:, :, 0], axis=1)
            S_dy = np.diff(S[:, :, 0], axis=0)
            E_smooth = np.sum(S_dx ** 2) + np.sum(S_dy ** 2)
            
            # 反射稀疏能量
            E_sparse = np.sum(np.abs(R))
            
            # 总能量
            E_total = self.lambda_recon * E_recon + \
                     self.lambda_smooth * E_smooth + \
                     self.lambda_sparse * E_sparse
            
            return E_total
        
        # 初始化参数
        init_params = np.concatenate([
            intrinsic.flatten(),
            shading.flatten(),
            reflection.flatten()
        ])
        
        # 优化
        from scipy.optimize import minimize
        result = minimize(
            total_energy,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 20}
        )
        
        # 解包结果
        n_pixels = H * W
        optimized_intrinsic = result.x[:n_pixels * C].reshape(H, W, C)
        optimized_shading = result.x[n_pixels * C:n_pixels * (C + 1)].reshape(H, W, 1)
        optimized_reflection = result.x[n_pixels * (C + 1):].reshape(H, W, C)
        
        # 归一化
        optimized_intrinsic = np.clip(optimized_intrinsic, 0, 1)
        optimized_shading = np.clip(optimized_shading, 0, 1)
        optimized_reflection = np.clip(optimized_reflection, 0, 1)
        
        return optimized_intrinsic, optimized_shading, optimized_reflection