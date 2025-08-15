import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import yaml
from tensorboardX import SummaryWriter

# 导入自定义模块
from preprocessing import AdaptiveHighlightDetector
from matrix_decomposition import WeightedSVT
from cyclegan_network import IntrinsicDecompositionNet, Discriminator, HighlightRemovalNet
from temporal_consistency import TemporalConsistencyLoss
from postprocessing import IterativeRefinement


class EndoscopeDataset(Dataset):
    """内窥镜图像数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 scale_factor: float = 0.5,
                 sequence_length: int = 3):
        """
        Args:
            data_dir: 数据目录
            transform: 数据变换
            scale_factor: 缩放因子
            sequence_length: 序列长度（用于时间一致性）
        """
        self.data_dir = data_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.sequence_length = sequence_length
        
        # 获取所有图像文件
        self.image_files = []
        for ext in ['*.jpg', '*.png', '*.bmp']:
            import glob
            self.image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
    def __len__(self):
        return max(0, len(self.image_files) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        # 加载图像序列
        sequence = []
        for i in range(self.sequence_length):
            img_path = self.image_files[idx + i]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 缩放
            if self.scale_factor != 1.0:
                h, w = img.shape[:2]
                new_h = int(h * self.scale_factor)
                new_w = int(w * self.scale_factor)
                img = cv2.resize(img, (new_w, new_h))
            
            # 归一化到[0, 1]
            img = img.astype(np.float32) / 255.0
            
            if self.transform:
                img = self.transform(img)
            
            sequence.append(img)
        
        # 转换为tensor
        sequence = [torch.from_numpy(img).permute(2, 0, 1) for img in sequence]
        
        return torch.stack(sequence)


class CycleGANTrainer:
    """CycleGAN训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化网络
        self.init_networks()
        
        # 初始化损失函数
        self.init_losses()
        
        # 初始化优化器
        self.init_optimizers()
        
        # 初始化其他组件
        self.highlight_detector = AdaptiveHighlightDetector()
        self.matrix_decomposer = WeightedSVT(rank=config['svt_rank'])
        self.postprocessor = IterativeRefinement(num_iterations=3)
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
    def init_networks(self):
        """初始化网络"""
        # 内在分解网络
        self.decomposition_net = IntrinsicDecompositionNet(
            input_channels=3
        ).to(self.device)
        
        # 高光去除网络
        self.highlight_removal_net = HighlightRemovalNet(
            input_channels=4  # RGB + mask
        ).to(self.device)
        
        # 判别器
        self.discriminator_intrinsic = Discriminator(
            input_channels=3
        ).to(self.device)
        
        self.discriminator_highlight = Discriminator(
            input_channels=3
        ).to(self.device)
        
    def init_losses(self):
        """初始化损失函数"""
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        
    def init_optimizers(self):
        """初始化优化器"""
        # 生成器优化器
        gen_params = list(self.decomposition_net.parameters()) + \
                    list(self.highlight_removal_net.parameters())
        self.optimizer_G = optim.Adam(
            gen_params,
            lr=self.config['lr_generator'],
            betas=(0.5, 0.999)
        )
        
        # 判别器优化器
        disc_params = list(self.discriminator_intrinsic.parameters()) + \
                     list(self.discriminator_highlight.parameters())
        self.optimizer_D = optim.Adam(
            disc_params,
            lr=self.config['lr_discriminator'],
            betas=(0.5, 0.999)
        )
        
    def train_step(self, batch: torch.Tensor, step: int) -> Dict[str, float]:
        """单个训练步骤"""
        batch = batch.to(self.device)
        batch_size, seq_len, C, H, W = batch.shape
        
        losses = {}
        
        # 提取中间帧作为主要处理对象
        main_frame = batch[:, seq_len // 2]
        
        # 步骤1: 高光检测
        highlight_masks = []
        for b in range(batch_size):
            frame_np = main_frame[b].cpu().numpy().transpose(1, 2, 0)
            mask, _ = self.highlight_detector.process(
                (frame_np * 255).astype(np.uint8),
                scale_factor=1.0
            )
            highlight_masks.append(torch.from_numpy(mask).unsqueeze(0))
        
        highlight_masks = torch.stack(highlight_masks).to(self.device)
        
        # 步骤2: 初始矩阵分解
        initial_decompositions = []
        for b in range(batch_size):
            frame_np = main_frame[b].cpu().numpy().transpose(1, 2, 0)
            mask_np = highlight_masks[b, 0].cpu().numpy()
            
            intrinsic_shading, reflection = self.matrix_decomposer.decompose(
                frame_np, mask_np
            )
            
            initial_decompositions.append({
                'intrinsic_shading': torch.from_numpy(intrinsic_shading).permute(2, 0, 1),
                'reflection': torch.from_numpy(reflection).permute(2, 0, 1)
            })
        
        # 步骤3: 网络精炼
        # 内在分解
        intrinsic, shading, reflection = self.decomposition_net(main_frame)
        
        # 高光去除
        input_with_mask = torch.cat([main_frame, highlight_masks], dim=1)
        refined_image = self.highlight_removal_net(input_with_mask)
        
        # 计算损失
        # 重构损失
        reconstruction = intrinsic * shading + reflection
        losses['recon'] = self.l1_loss(reconstruction, main_frame)
        
        # 初始分解指导损失
        if len(initial_decompositions) > 0:
            init_intrinsic_shading = torch.stack([
                d['intrinsic_shading'] for d in initial_decompositions
            ]).to(self.device)
            init_reflection = torch.stack([
                d['reflection'] for d in initial_decompositions
            ]).to(self.device)
            
            losses['guide_intrinsic'] = self.l1_loss(
                intrinsic * shading, init_intrinsic_shading
            )
            losses['guide_reflection'] = self.l1_loss(
                reflection, init_reflection
            )
        
        # 稀疏性损失（反射应该稀疏）
        losses['sparse'] = torch.mean(torch.abs(reflection))
        
        # 平滑性损失（光照应该平滑）
        shading_dx = torch.abs(shading[:, :, :, 1:] - shading[:, :, :, :-1])
        shading_dy = torch.abs(shading[:, :, 1:, :] - shading[:, :, :-1, :])
        losses['smooth'] = torch.mean(shading_dx) + torch.mean(shading_dy)
        
        # 时间一致性损失（如果有多帧）
        if seq_len > 1:
            all_intrinsics = []
            all_frames = []
            for t in range(seq_len):
                frame_t = batch[:, t]
                intrinsic_t, _, _ = self.decomposition_net(frame_t)
                all_intrinsics.append(intrinsic_t)
                all_frames.append(frame_t)
            
            losses['temporal'] = self.temporal_loss(all_intrinsics, all_frames)
        
        # GAN损失
        # 训练判别器
        self.optimizer_D.zero_grad()
        
        # 判别内在颜色
        real_score = self.discriminator_intrinsic(intrinsic.detach())
        # 根据实际输出尺寸创建标签
        real_labels = torch.ones_like(real_score).to(self.device)
        fake_labels = torch.zeros_like(real_score).to(self.device)
        
        d_loss_real = self.gan_loss(real_score, real_labels)
        
        # 这里应该有真实的无高光图像，暂时用refined_image
        fake_score = self.discriminator_intrinsic(refined_image.detach())
        d_loss_fake = self.gan_loss(fake_score, fake_labels)
        
        losses['d_loss'] = d_loss_real + d_loss_fake
        losses['d_loss'].backward()
        self.optimizer_D.step()
        
        # 训练生成器
        self.optimizer_G.zero_grad()
        
        # 生成器的GAN损失
        gen_score = self.discriminator_intrinsic(intrinsic)
        losses['g_loss'] = self.gan_loss(gen_score, real_labels)
        
        # 总损失
        total_loss = (losses['recon'] * self.config['lambda_recon'] +
                     losses.get('guide_intrinsic', 0) * self.config['lambda_guide'] +
                     losses.get('guide_reflection', 0) * self.config['lambda_guide'] +
                     losses['sparse'] * self.config['lambda_sparse'] +
                     losses['smooth'] * self.config['lambda_smooth'] +
                     losses.get('temporal', 0) * self.config['lambda_temporal'] +
                     losses['g_loss'] * self.config['lambda_gan'])
        
        total_loss.backward()
        self.optimizer_G.step()
        
        # 清理显存缓存
        torch.cuda.empty_cache()
            
        # 记录损失
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return loss_dict
    
    def train(self, train_loader: DataLoader, num_epochs: int):
        """训练循环"""
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_losses = {}
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    losses = self.train_step(batch, global_step)
                    
                    # 累积损失
                    for k, v in losses.items():
                        if k not in epoch_losses:
                            epoch_losses[k] = []
                        epoch_losses[k].append(v)
                    
                    # 更新进度条
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
                    
                    # 记录到TensorBoard
                    if global_step % self.config['log_interval'] == 0:
                        for k, v in losses.items():
                            self.writer.add_scalar(f'Loss/{k}', v, global_step)
                    
                    global_step += 1
            
            # 打印epoch统计
            print(f"\nEpoch {epoch+1} Summary:")
            for k, v in epoch_losses.items():
                avg_loss = np.mean(v)
                print(f"  {k}: {avg_loss:.4f}")
                self.writer.add_scalar(f'Epoch/{k}', avg_loss, epoch)
            
            # 保存检查点
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'decomposition_net': self.decomposition_net.state_dict(),
            'highlight_removal_net': self.highlight_removal_net.state_dict(),
            'discriminator_intrinsic': self.discriminator_intrinsic.state_dict(),
            'discriminator_highlight': self.discriminator_highlight.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }
        
        save_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train image decomposition network')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    
    args = parser.parse_args()
    
    # 默认配置
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'lr_generator': args.lr,
        'lr_discriminator': args.lr,
        'svt_rank': 10,
        'lambda_recon': 1.0,
        'lambda_guide': 0.5,
        'lambda_sparse': 0.1,
        'lambda_smooth': 0.1,
        'lambda_temporal': 0.5,
        'lambda_gan': 0.1,
        'log_interval': 10,
        'save_interval': 10,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'scale_factor': 0.5,
        'sequence_length': 3
    }
    
    # 如果有配置文件，加载它
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # 创建必要的目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset = EndoscopeDataset(
        data_dir=config['data_dir'],
        scale_factor=config['scale_factor'],
        sequence_length=config['sequence_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        
    )
    
    # 创建训练器
    trainer = CycleGANTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, config['num_epochs'])


if __name__ == '__main__':
    main()