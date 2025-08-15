import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from preprocessing import AdaptiveHighlightDetector
from matrix_decomposition import WeightedSVT, RobustPCA
from cyclegan_network import IntrinsicDecompositionNet, HighlightRemovalNet
from postprocessing import IterativeRefinement
from temporal_consistency import MultiFrameProcessor, TemporalSmoothing


class ImageDecomposer:
    """完整的图像分解推理系统"""
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda',
                 use_temporal_consistency: bool = True):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备 ('cuda' or 'cpu')
            use_temporal_consistency: 是否使用时间一致性
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_temporal_consistency = use_temporal_consistency
        
        # 初始化各个组件
        self.highlight_detector = AdaptiveHighlightDetector()
        self.matrix_decomposer = WeightedSVT(rank=10)
        self.postprocessor = IterativeRefinement(num_iterations=3)
        
        # 初始化网络
        self.decomposition_net = IntrinsicDecompositionNet().to(self.device)
        self.highlight_removal_net = HighlightRemovalNet().to(self.device)
        
        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("No checkpoint loaded, using random initialization")
        
        # 时间一致性组件
        if use_temporal_consistency:
            self.frame_processor = MultiFrameProcessor(window_size=5)
            self.temporal_smoother = TemporalSmoothing(alpha=0.8)
        
        # 设置为评估模式
        self.decomposition_net.eval()
        self.highlight_removal_net.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'decomposition_net' in checkpoint:
            self.decomposition_net.load_state_dict(checkpoint['decomposition_net'])
        if 'highlight_removal_net' in checkpoint:
            self.highlight_removal_net.load_state_dict(checkpoint['highlight_removal_net'])
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 640)) -> np.ndarray:
        """预处理图像"""
        # 调整大小
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # 归一化到[0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def decompose_single_frame(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        分解单帧图像
        
        Args:
            image: 输入图像 (H, W, 3) RGB格式，值范围[0, 255]或[0, 1]
            
        Returns:
            dict: 包含'intrinsic', 'shading', 'reflection', 'highlight_mask'
        """
        # 预处理
        image = self.preprocess_image(image)
        H, W = image.shape[:2]
        
        # 步骤1: 高光检测
        highlight_mask, scaled_image = self.highlight_detector.process(
            (image * 255).astype(np.uint8) if image.max() <= 1 else image,
            scale_factor=1.0
        )
        
        # 步骤2: 初始矩阵分解
        initial_intrinsic_shading, initial_reflection = self.matrix_decomposer.decompose(
            image, highlight_mask
        )
        
        # 步骤3: 神经网络精炼
        with torch.no_grad():
            # 准备输入
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            mask_tensor = torch.from_numpy(highlight_mask).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # 内在分解
            intrinsic, shading, reflection = self.decomposition_net(image_tensor)
            
            # 高光去除（可选）
            input_with_mask = torch.cat([image_tensor, mask_tensor], dim=1)
            refined_no_highlight = self.highlight_removal_net(input_with_mask)
        
        # 转换回numpy
        intrinsic_np = intrinsic[0].cpu().numpy().transpose(1, 2, 0)
        shading_np = shading[0].cpu().numpy().transpose(1, 2, 0)
        reflection_np = reflection[0].cpu().numpy().transpose(1, 2, 0)
        
        # 步骤4: 后处理优化
        refined_results = self.postprocessor.refine(
            image,
            intrinsic_np,
            shading_np,
            reflection_np,
            highlight_mask
        )
        
        # 添加额外信息
        refined_results['highlight_mask'] = highlight_mask
        refined_results['refined_image'] = refined_no_highlight[0].cpu().numpy().transpose(1, 2, 0)
        refined_results['original'] = image
        
        return refined_results
    
    def decompose_video(self, frames: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        分解视频序列，保持时间一致性
        
        Args:
            frames: 视频帧列表
            
        Returns:
            分解结果列表
        """
        results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # 单帧分解
            frame_result = self.decompose_single_frame(frame)
            
            if self.use_temporal_consistency:
                # 添加到缓冲区
                self.frame_processor.add_frame(frame, frame_result['intrinsic'])
                
                # 应用时间平滑
                smoothed_intrinsic = self.temporal_smoother.smooth(frame_result['intrinsic'])
                frame_result['intrinsic'] = smoothed_intrinsic
                
                # 使用时间中值滤波
                median_intrinsic = self.frame_processor.temporal_median_filter()
                if median_intrinsic is not None:
                    # 混合当前结果和中值滤波结果
                    frame_result['intrinsic'] = 0.7 * frame_result['intrinsic'] + 0.3 * median_intrinsic
            
            results.append(frame_result)
        
        return results
    
    def visualize_decomposition(self, result: Dict[str, np.ndarray], save_path: Optional[str] = None):
        """可视化分解结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(result['original'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 内在颜色
        axes[0, 1].imshow(result['intrinsic'])
        axes[0, 1].set_title("Intrinsic Color (I')")
        axes[0, 1].axis('off')
        
        # 光照
        shading_vis = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
        axes[0, 2].imshow(shading_vis, cmap='gray' if result['shading'].shape[2] == 1 else None)
        axes[0, 2].set_title('Shading (S)')
        axes[0, 2].axis('off')
        
        # 反射
        axes[1, 0].imshow(result['reflection'])
        axes[1, 0].set_title('Reflection (R)')
        axes[1, 0].axis('off')
        
        # 高光mask
        axes[1, 1].imshow(result['highlight_mask'], cmap='hot')
        axes[1, 1].set_title('Highlight Mask')
        axes[1, 1].axis('off')
        
        # 重构
        reconstruction = result['intrinsic'] * (np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']) + result['reflection']
        axes[1, 2].imshow(np.clip(reconstruction, 0, 1))
        axes[1, 2].set_title('Reconstruction')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def process_image(image_path: str, 
                  output_dir: str,
                  checkpoint_path: Optional[str] = None,
                  visualize: bool = True):
    """处理单张图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建分解器
    decomposer = ImageDecomposer(checkpoint_path=checkpoint_path)
    
    # 分解图像
    print(f"Processing {image_path}...")
    result = decomposer.decompose_single_frame(image)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存各个分量
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_intrinsic.png"), 
                cv2.cvtColor((result['intrinsic'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    shading_3ch = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_shading.png"),
                (shading_3ch * 255).astype(np.uint8))
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_reflection.png"),
                cv2.cvtColor((result['reflection'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_highlight_mask.png"),
                (result['highlight_mask'] * 255).astype(np.uint8))
    
    # 可视化
    if visualize:
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        decomposer.visualize_decomposition(result, save_path=vis_path)
    
    print(f"Results saved to {output_dir}")
    
    return result


def process_video(video_path: str,
                 output_dir: str,
                 checkpoint_path: Optional[str] = None,
                 max_frames: int = -1):
    """处理视频"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 读取所有帧
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        
        if max_frames > 0 and frame_count >= max_frames:
            break
    
    cap.release()
    print(f"Loaded {len(frames)} frames from video")
    
    # 创建分解器
    decomposer = ImageDecomposer(checkpoint_path=checkpoint_path, use_temporal_consistency=True)
    
    # 处理视频
    results = decomposer.decompose_video(frames)
    
    # 保存结果视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 创建不同的输出视频
    writers = {
        'intrinsic': cv2.VideoWriter(os.path.join(output_dir, 'intrinsic.mp4'), fourcc, fps, (width, height)),
        'shading': cv2.VideoWriter(os.path.join(output_dir, 'shading.mp4'), fourcc, fps, (width, height)),
        'reflection': cv2.VideoWriter(os.path.join(output_dir, 'reflection.mp4'), fourcc, fps, (width, height)),
        'reconstruction': cv2.VideoWriter(os.path.join(output_dir, 'reconstruction.mp4'), fourcc, fps, (width, height))
    }
    
    for result in tqdm(results, desc="Saving videos"):
        # 内在颜色
        intrinsic_bgr = cv2.cvtColor((result['intrinsic'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writers['intrinsic'].write(intrinsic_bgr)
        
        # 光照
        shading_3ch = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
        shading_bgr = (shading_3ch * 255).astype(np.uint8)
        writers['shading'].write(shading_bgr)
        
        # 反射
        reflection_bgr = cv2.cvtColor((result['reflection'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writers['reflection'].write(reflection_bgr)
        
        # 重构
        reconstruction = result['intrinsic'] * shading_3ch + result['reflection']
        reconstruction_bgr = cv2.cvtColor((np.clip(reconstruction, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writers['reconstruction'].write(reconstruction_bgr)
    
    # 释放所有写入器
    for writer in writers.values():
        writer.release()
    
    print(f"Video results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Image decomposition inference')
    parser.add_argument('--input', type=str, required=True, help='Input image or video path')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--type', type=str, choices=['image', 'video'], default='image', help='Input type')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--max_frames', type=int, default=-1, help='Maximum frames to process for video')
    
    args = parser.parse_args()
    
    if args.type == 'image':
        process_image(args.input, args.output, args.checkpoint, args.visualize)
    else:
        process_video(args.input, args.output, args.checkpoint, args.max_frames)


if __name__ == '__main__':
    main()