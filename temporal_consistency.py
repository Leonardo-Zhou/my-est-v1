import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2


class TemporalConsistencyLoss(nn.Module):
    """多帧时间连续性损失"""
    
    def __init__(self, 
                 motion_weight: float = 1.0,
                 color_weight: float = 1.0,
                 structure_weight: float = 0.5):
        super(TemporalConsistencyLoss, self).__init__()
        self.motion_weight = motion_weight
        self.color_weight = color_weight
        self.structure_weight = structure_weight
        
    def compute_optical_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算光流用于运动对齐"""
        # 转换为numpy进行光流计算
        f1_np = frame1.cpu().numpy().transpose(0, 2, 3, 1)
        f2_np = frame2.cpu().numpy().transpose(0, 2, 3, 1)
        
        batch_size = f1_np.shape[0]
        flows = []
        
        for i in range(batch_size):
            gray1 = cv2.cvtColor((f1_np[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor((f2_np[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
        
        flows = np.stack(flows)
        flow_tensor = torch.from_numpy(flows).permute(0, 3, 1, 2).to(frame1.device)
        
        return flow_tensor[:, 0:1, :, :], flow_tensor[:, 1:2, :, :]
    
    def warp_image(self, image: torch.Tensor, flow_x: torch.Tensor, flow_y: torch.Tensor) -> torch.Tensor:
        """使用光流扭曲图像"""
        B, C, H, W = image.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=torch.float32),
            torch.arange(W, device=image.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 应用光流
        new_x = grid_x.unsqueeze(0) + flow_x.squeeze(1)
        new_y = grid_y.unsqueeze(0) + flow_y.squeeze(1)
        
        # 归一化到[-1, 1]
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0
        
        # 堆叠并重排
        grid = torch.stack([new_x, new_y], dim=-1)
        
        # 扭曲图像
        warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def forward(self, 
                intrinsics: List[torch.Tensor],
                frames: List[torch.Tensor]) -> torch.Tensor:
        """
        计算多帧内在颜色的连续性损失
        
        Args:
            intrinsics: 多帧的内在颜色 [I'_t, I'_t+1, ...]
            frames: 原始帧 [I_t, I_t+1, ...]
            
        Returns:
            consistency_loss: 连续性损失
        """
        if len(intrinsics) < 2:
            return torch.tensor(0.0, device=intrinsics[0].device)
        
        total_loss = 0.0
        num_pairs = len(intrinsics) - 1
        
        for i in range(num_pairs):
            # 当前帧和下一帧
            I_prime_t = intrinsics[i]
            I_prime_next = intrinsics[i + 1]
            frame_t = frames[i]
            frame_next = frames[i + 1]
            
            # 计算光流
            flow_x, flow_y = self.compute_optical_flow(frame_t, frame_next)
            
            # 扭曲当前帧的内在颜色到下一帧
            I_prime_t_warped = self.warp_image(I_prime_t, flow_x, flow_y)
            
            # 颜色一致性损失
            color_loss = F.l1_loss(I_prime_t_warped, I_prime_next)
            
            # 结构相似性损失
            structure_loss = 1 - self.compute_ssim(I_prime_t_warped, I_prime_next)
            
            # 梯度一致性损失（边缘保持）
            grad_loss = self.gradient_consistency_loss(I_prime_t_warped, I_prime_next)
            
            # 组合损失
            pair_loss = (self.color_weight * color_loss + 
                        self.structure_weight * structure_loss +
                        self.motion_weight * grad_loss)
            
            total_loss += pair_loss
        
        return total_loss / num_pairs
    
    def compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算结构相似性"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, padding=1)
        mu_y = F.avg_pool2d(y, 3, 1, padding=1)
        
        sigma_x = F.avg_pool2d(x ** 2, 3, 1, padding=1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, padding=1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim.mean()
    
    def gradient_consistency_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """梯度一致性损失"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        # 计算梯度
        grad_x_1 = F.conv2d(x.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y_1 = F.conv2d(x.mean(dim=1, keepdim=True), sobel_y, padding=1)
        grad_x_2 = F.conv2d(y.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y_2 = F.conv2d(y.mean(dim=1, keepdim=True), sobel_y, padding=1)
        
        # 梯度差异
        grad_diff_x = F.l1_loss(grad_x_1, grad_x_2)
        grad_diff_y = F.l1_loss(grad_y_1, grad_y_2)
        
        return grad_diff_x + grad_diff_y


class MultiFrameProcessor:
    """多帧处理器，确保内在颜色的连续性"""
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 处理窗口大小
        """
        self.window_size = window_size
        self.frame_buffer = []
        self.intrinsic_buffer = []
        
    def add_frame(self, frame: np.ndarray, intrinsic: np.ndarray):
        """添加新帧到缓冲区"""
        self.frame_buffer.append(frame)
        self.intrinsic_buffer.append(intrinsic)
        
        # 保持窗口大小
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
            self.intrinsic_buffer.pop(0)
    
    def temporal_median_filter(self) -> np.ndarray:
        """时间中值滤波，增强内在颜色的稳定性"""
        if len(self.intrinsic_buffer) == 0:
            return None
        
        if len(self.intrinsic_buffer) == 1:
            return self.intrinsic_buffer[0]
        
        # 堆叠所有内在颜色
        stacked = np.stack(self.intrinsic_buffer, axis=0)
        
        # 计算中值
        median_intrinsic = np.median(stacked, axis=0)
        
        return median_intrinsic
    
    def weighted_temporal_average(self, weights: Optional[List[float]] = None) -> np.ndarray:
        """加权时间平均"""
        if len(self.intrinsic_buffer) == 0:
            return None
        
        if weights is None:
            # 默认权重：越新的帧权重越大
            weights = np.exp(np.linspace(-1, 0, len(self.intrinsic_buffer)))
            weights = weights / weights.sum()
        
        # 加权平均
        weighted_sum = np.zeros_like(self.intrinsic_buffer[0])
        for i, intrinsic in enumerate(self.intrinsic_buffer):
            weighted_sum += weights[i] * intrinsic
        
        return weighted_sum
    
    def kalman_filter_update(self, new_intrinsic: np.ndarray, 
                             process_noise: float = 0.01,
                             measurement_noise: float = 0.1) -> np.ndarray:
        """使用卡尔曼滤波更新内在颜色估计"""
        if not hasattr(self, 'kalman_state'):
            # 初始化卡尔曼滤波器状态
            self.kalman_state = new_intrinsic.copy()
            self.kalman_covariance = np.ones_like(new_intrinsic) * 1.0
            return new_intrinsic
        
        # 预测步骤
        predicted_state = self.kalman_state
        predicted_covariance = self.kalman_covariance + process_noise
        
        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise)
        self.kalman_state = predicted_state + kalman_gain * (new_intrinsic - predicted_state)
        self.kalman_covariance = (1 - kalman_gain) * predicted_covariance
        
        return self.kalman_state


class SpatioTemporalRegularizer:
    """时空正则化器"""
    
    def __init__(self, spatial_weight: float = 0.5, temporal_weight: float = 0.5):
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        
    def regularize(self, 
                  intrinsics: List[np.ndarray],
                  frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        对多帧内在颜色进行时空正则化
        
        Args:
            intrinsics: 内在颜色序列
            frames: 原始帧序列
            
        Returns:
            regularized_intrinsics: 正则化后的内在颜色
        """
        num_frames = len(intrinsics)
        if num_frames < 2:
            return intrinsics
        
        regularized = []
        
        for t in range(num_frames):
            # 空间正则化（双边滤波）
            spatial_reg = cv2.bilateralFilter(
                (intrinsics[t] * 255).astype(np.uint8),
                d=9, sigmaColor=75, sigmaSpace=75
            ) / 255.0
            
            # 时间正则化
            temporal_reg = intrinsics[t].copy()
            
            if t > 0:
                # 与前一帧的一致性
                flow = self._compute_flow(frames[t-1], frames[t])
                warped_prev = self._warp_image(intrinsics[t-1], flow)
                temporal_reg = 0.7 * temporal_reg + 0.3 * warped_prev
            
            if t < num_frames - 1:
                # 与后一帧的一致性
                flow = self._compute_flow(frames[t+1], frames[t])
                warped_next = self._warp_image(intrinsics[t+1], flow)
                temporal_reg = 0.7 * temporal_reg + 0.3 * warped_next
            
            # 组合空间和时间正则化
            regularized_intrinsic = (self.spatial_weight * spatial_reg + 
                                    self.temporal_weight * temporal_reg)
            
            regularized.append(regularized_intrinsic)
        
        return regularized
    
    def _compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """计算光流"""
        gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        return flow
    
    def _warp_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """使用光流扭曲图像"""
        h, w = flow.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                flow_map[y, x] = [x + flow[y, x, 0], y + flow[y, x, 1]]
        
        warped = cv2.remap(image, flow_map[:, :, 0], flow_map[:, :, 1],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped


class TemporalSmoothing:
    """时间平滑模块"""
    
    def __init__(self, alpha: float = 0.8):
        """
        Args:
            alpha: 平滑因子 (0-1)，越大越平滑
        """
        self.alpha = alpha
        self.prev_intrinsic = None
        
    def smooth(self, current_intrinsic: np.ndarray) -> np.ndarray:
        """应用时间平滑"""
        if self.prev_intrinsic is None:
            self.prev_intrinsic = current_intrinsic
            return current_intrinsic
        
        # 指数移动平均
        smoothed = self.alpha * self.prev_intrinsic + (1 - self.alpha) * current_intrinsic
        
        # 更新历史
        self.prev_intrinsic = smoothed
        
        return smoothed
    
    def reset(self):
        """重置平滑器状态"""
        self.prev_intrinsic = None