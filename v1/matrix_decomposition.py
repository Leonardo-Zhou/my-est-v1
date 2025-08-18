import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
from typing import Tuple, Optional
import cv2


class SVTDecomposition:
    """奇异值阈值分解 (Singular Value Thresholding) 用于初始图像分解"""
    
    def __init__(self, 
                 rank: Optional[int] = None,
                 threshold: float = 0.1,
                 max_iter: int = 100,
                 tol: float = 1e-4):
        """
        Args:
            rank: 低秩近似的秩，None表示自动确定
            threshold: 奇异值阈值
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.rank = rank
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol
        
    def soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """软阈值操作"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def svt_operator(self, matrix: np.ndarray, threshold: float) -> np.ndarray:
        """奇异值阈值算子"""
        U, s, Vt = svd(matrix, full_matrices=False)
        
        # 对奇异值应用软阈值
        s_thresh = self.soft_threshold(s, threshold)
        
        # 如果指定了秩，只保留前rank个奇异值
        if self.rank is not None:
            s_thresh[self.rank:] = 0
            
        # 重构矩阵
        return U @ np.diag(s_thresh) @ Vt
    
    def decompose(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将图像分解为低秩部分（I'*S）和稀疏部分（R）
        
        Args:
            image: 输入图像 (H, W, C)
            mask: 高光区域mask (H, W)，可选
            
        Returns:
            low_rank: 低秩部分 I'*S (H, W, C)
            sparse: 稀疏部分 R (H, W, C)
        """
        H, W, C = image.shape
        
        # 将图像重塑为矩阵形式 (H*W, C)
        image_matrix = image.reshape(-1, C)
        
        # 初始化
        low_rank_matrix = np.zeros_like(image_matrix)
        sparse_matrix = np.zeros_like(image_matrix)
        
        # 如果提供了mask，使用它来引导分解
        if mask is not None:
            mask_flat = mask.flatten()
            weight = 1 + mask_flat[:, np.newaxis] * 4  # 高光区域权重更高
        else:
            weight = np.ones_like(image_matrix)
        
        # 对每个颜色通道进行分解
        for c in range(C):
            channel = image_matrix[:, c].reshape(H, W)
            
            # 初始化
            L = np.zeros_like(channel)  # 低秩部分
            S = np.zeros_like(channel)  # 稀疏部分
            Y = np.zeros_like(channel)  # 拉格朗日乘数
            
            mu = 1e-3
            rho = 1.1
            
            for iter in range(self.max_iter):
                L_old = L.copy()
                
                # 更新L（低秩部分）- 使用SVT
                L = self.svt_operator(channel - S + Y/mu, 1/mu)
                
                # 更新S（稀疏部分）- 使用软阈值
                S = self.soft_threshold(channel - L + Y/mu, self.threshold/mu)
                
                # 如果有mask，约束稀疏部分主要在高光区域
                if mask is not None:
                    S = S * mask
                
                # 更新拉格朗日乘数
                Y = Y + mu * (channel - L - S)
                
                # 更新惩罚参数
                mu = min(mu * rho, 1e10)
                
                # 检查收敛
                if np.linalg.norm(L - L_old, 'fro') < self.tol:
                    break
            
            low_rank_matrix[:, c] = L.flatten()
            sparse_matrix[:, c] = S.flatten()
        
        # 重塑回图像形状
        low_rank = low_rank_matrix.reshape(H, W, C)
        sparse = sparse_matrix.reshape(H, W, C)
        
        # 确保值在有效范围内
        low_rank = np.clip(low_rank, 0, 1)
        sparse = np.clip(sparse, 0, 1)
        
        return low_rank, sparse


class RobustPCA:
    """鲁棒主成分分析用于图像分解"""
    
    def __init__(self, lambda_param: float = None, mu: float = None, 
                 max_iter: int = 500, tol: float = 1e-7):
        """
        Args:
            lambda_param: 稀疏正则化参数
            mu: 增广拉格朗日参数
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.lambda_param = lambda_param
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
    
    def decompose(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用RPCA将图像分解
        
        Args:
            image: 输入图像 (H, W, C)
            mask: 高光mask (H, W)
            
        Returns:
            low_rank: 低秩部分 (H, W, C)
            sparse: 稀疏部分 (H, W, C)
        """
        H, W, C = image.shape
        
        # 设置默认参数
        if self.lambda_param is None:
            self.lambda_param = 1.0 / np.sqrt(max(H, W))
        if self.mu is None:
            self.mu = 10 * self.lambda_param
            
        # 初始化
        low_rank = np.zeros_like(image)
        sparse = np.zeros_like(image)
        
        for c in range(C):
            M = image[:, :, c]
            
            # 初始化变量
            L = np.zeros_like(M)  # 低秩矩阵
            S = np.zeros_like(M)  # 稀疏矩阵
            Y = np.zeros_like(M)  # 对偶变量
            
            mu = self.mu
            mu_inv = 1.0 / mu
            
            for iteration in range(self.max_iter):
                # 更新L - 使用SVD
                U, sigma, Vt = np.linalg.svd(M - S + mu_inv * Y, full_matrices=False)
                sigma_thresh = np.maximum(sigma - mu_inv, 0)
                L = U @ np.diag(sigma_thresh) @ Vt
                
                # 更新S - 软阈值
                S_temp = M - L + mu_inv * Y
                S = np.sign(S_temp) * np.maximum(np.abs(S_temp) - self.lambda_param * mu_inv, 0)
                
                # 如果有mask，约束稀疏部分
                if mask is not None:
                    S = S * mask
                
                # 更新Y
                Y = Y + mu * (M - L - S)
                
                # 检查收敛
                err = np.linalg.norm(M - L - S, 'fro')
                if err < self.tol:
                    break
                    
            low_rank[:, :, c] = L
            sparse[:, :, c] = S
        
        # 归一化到[0, 1]
        low_rank = np.clip(low_rank, 0, 1)
        sparse = np.clip(sparse, 0, 1)
        
        return low_rank, sparse


class WeightedSVT:
    """加权SVT分解，考虑高光区域的权重"""
    
    def __init__(self, rank: int = 10, weight_factor: float = 5.0):
        self.rank = rank
        self.weight_factor = weight_factor
        
    def decompose(self, image: np.ndarray, highlight_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用加权SVT进行分解
        
        Args:
            image: 输入图像 (H, W, C)
            highlight_mask: 高光区域mask (H, W)
            
        Returns:
            intrinsic_shading: I'*S部分 (H, W, C)
            reflection: R部分 (H, W, C)
        """
        H, W, C = image.shape
        
        # 创建权重矩阵
        weights = 1.0 + highlight_mask * self.weight_factor
        
        intrinsic_shading = np.zeros_like(image)
        reflection = np.zeros_like(image)
        
        for c in range(C):
            channel = image[:, :, c]
            
            # 加权SVD分解
            weighted_channel = channel * np.sqrt(weights)
            U, s, Vt = np.linalg.svd(weighted_channel, full_matrices=False)
            
            # 保留前rank个分量作为低秩部分
            s_truncated = s.copy()
            s_truncated[self.rank:] = 0
            
            # 重构低秩部分
            low_rank_weighted = U @ np.diag(s_truncated) @ Vt
            low_rank = low_rank_weighted / np.sqrt(weights + 1e-8)
            
            # 稀疏部分为残差
            sparse = channel - low_rank
            
            # 在高光区域增强稀疏部分
            sparse = sparse * (1 + highlight_mask * 2)
            
            intrinsic_shading[:, :, c] = low_rank
            reflection[:, :, c] = sparse
            
        # 后处理：确保反射主要在高光区域
        reflection = reflection * np.expand_dims(highlight_mask, -1)
        
        # 重新计算intrinsic_shading以保证重构
        intrinsic_shading = image - reflection
        
        # 归一化
        intrinsic_shading = np.clip(intrinsic_shading, 0, 1)
        reflection = np.clip(reflection, 0, 1)
        
        return intrinsic_shading, reflection