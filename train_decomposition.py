import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
from typing import Dict, List, Tuple, Optional
import argparse
import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 导入分解相关模块
from endosrr_decomposition import EndoSRRBasedDecomposition
from improved_decomposition import PhysicallyBasedDecomposition


class DecompositionNet(nn.Module):
    """
    端到端的分解网络
    输入: 原始图像 I
    输出: 内在颜色 I', 光照 S, 反射 R
    约束: I = I' * S + R
    """
    
    def __init__(self, input_channels=3, base_features=64):
        super(DecompositionNet, self).__init__()
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_features, 7, padding=3),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features*2, base_features*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_features*4),
            nn.ReLU(inplace=True),
        )
        
        # 高光检测分支 (输出mask)
        self.highlight_branch = nn.Sequential(
            nn.Conv2d(base_features*4, base_features*2, 3, padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features*2, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 7, padding=3),
            nn.Sigmoid()  # 输出[0,1]的mask
        )
        
        # 内在颜色分支
        self.intrinsic_branch = nn.Sequential(
            nn.Conv2d(base_features*4, base_features*4, 3, padding=1),
            nn.BatchNorm2d(base_features*4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_features*4),
            ResidualBlock(base_features*4),
            nn.ConvTranspose2d(base_features*4, base_features*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features*2, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, 3, 7, padding=3),
            nn.Sigmoid()  # 内在颜色在[0,1]
        )
        
        # 光照分支 (输出单通道)
        self.shading_branch = nn.Sequential(
            nn.Conv2d(base_features*4, base_features*2, 3, padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_features*2),
            nn.ConvTranspose2d(base_features*2, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 7, padding=3),
            nn.Sigmoid()  # 光照在[0,1]
        )
        
        # 反射分支
        self.reflection_branch = nn.Sequential(
            nn.Conv2d(base_features*4 + 1, base_features*4, 3, padding=1),  # +1 for mask
            nn.BatchNorm2d(base_features*4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_features*4),
            nn.ConvTranspose2d(base_features*4, base_features*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features*2, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, 3, 7, padding=3),
            nn.Sigmoid()  # 反射在[0,1]
        )
    
    def forward(self, x):
        # 共享特征提取
        features = self.encoder(x)
        
        # 高光检测
        highlight_mask = self.highlight_branch(features)
        
        # 内在颜色
        intrinsic = self.intrinsic_branch(features)
        
        # 光照
        shading = self.shading_branch(features)
        
        # 反射（使用mask作为额外输入）
        features_with_mask = torch.cat([features, F.interpolate(highlight_mask, 
                                                                 size=features.shape[2:], 
                                                                 mode='bilinear', 
                                                                 align_corners=False)], dim=1)
        reflection = self.reflection_branch(features_with_mask)
        
        # 应用mask约束反射
        reflection = reflection * highlight_mask
        
        return intrinsic, shading, reflection, highlight_mask


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DecompositionLoss(nn.Module):
    """
    分解损失函数
    """
    
    def __init__(self, config):
        super(DecompositionLoss, self).__init__()
        self.lambda_recon = config.get('lambda_recon', 1.0)
        self.lambda_smooth = config.get('lambda_smooth', 0.1)
        self.lambda_sparse = config.get('lambda_sparse', 0.01)
        self.lambda_intrinsic = config.get('lambda_intrinsic', 0.1)
        self.lambda_mask = config.get('lambda_mask', 0.1)
        
    def reconstruction_loss(self, image, intrinsic, shading, reflection):
        """重构损失: I = I' * S + R"""
        shading_3ch = shading.repeat(1, 3, 1, 1)
        reconstruction = intrinsic * shading_3ch + reflection
        return F.l1_loss(reconstruction, image)
    
    def smoothness_loss(self, shading):
        """光照平滑损失"""
        # 计算梯度
        grad_x = torch.abs(shading[:, :, :, 1:] - shading[:, :, :, :-1])
        grad_y = torch.abs(shading[:, :, 1:, :] - shading[:, :, :-1, :])
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def sparsity_loss(self, reflection, mask):
        """反射稀疏损失"""
        # 反射应该主要在mask区域
        masked_reflection = reflection * mask
        return torch.mean(torch.abs(reflection - masked_reflection))
    
    def intrinsic_consistency_loss(self, intrinsic):
        """内在颜色一致性损失"""
        # 颜色应该相对均匀
        mean_color = torch.mean(intrinsic, dim=(2, 3), keepdim=True)
        return torch.mean((intrinsic - mean_color) ** 2)
    
    def mask_regularization_loss(self, mask):
        """Mask正则化损失"""
        # Mask应该相对稀疏但连续
        sparsity = torch.mean(mask)
        # 边缘平滑
        grad_x = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
        grad_y = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
        smoothness = torch.mean(grad_x) + torch.mean(grad_y)
        return sparsity + 0.1 * smoothness
    
    def forward(self, image, intrinsic, shading, reflection, mask):
        """计算总损失"""
        losses = {}
        
        # 重构损失（最重要）
        losses['reconstruction'] = self.reconstruction_loss(image, intrinsic, shading, reflection)
        
        # 光照平滑
        losses['smoothness'] = self.smoothness_loss(shading)
        
        # 反射稀疏
        losses['sparsity'] = self.sparsity_loss(reflection, mask)
        
        # 内在颜色一致性
        losses['intrinsic_consistency'] = self.intrinsic_consistency_loss(intrinsic)
        
        # Mask正则化
        losses['mask_regularization'] = self.mask_regularization_loss(mask)
        
        # 总损失
        total_loss = (self.lambda_recon * losses['reconstruction'] +
                     self.lambda_smooth * losses['smoothness'] +
                     self.lambda_sparse * losses['sparsity'] +
                     self.lambda_intrinsic * losses['intrinsic_consistency'] +
                     self.lambda_mask * losses['mask_regularization'])
        
        losses['total'] = total_loss
        
        return losses


class EndoscopeDecompositionDataset(Dataset):
    """
    内窥镜图像分解数据集
    支持无监督和弱监督训练
    """
    
    def __init__(self, 
                 image_dir: str,
                 mask_dir: Optional[str] = None,
                 transform=None,
                 image_size=(256, 320),
                 use_pseudo_labels=True):
        """
        Args:
            image_dir: 图像目录
            mask_dir: 可选的mask目录（弱监督）
            transform: 数据增强
            image_size: 目标大小 (H, W)
            use_pseudo_labels: 是否使用伪标签
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.use_pseudo_labels = use_pseudo_labels
        
        # 获取所有图像
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')) + 
                                 glob.glob(os.path.join(image_dir, '*.jpg')))
        
        print(f"Found {len(self.image_paths)} images")
        
        # 如果使用伪标签，初始化分解器
        if use_pseudo_labels:
            self.decomposer = EndoSRRBasedDecomposition(use_iterative=False)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image.astype(np.float32) / 255.0
        
        sample = {'image': image}
        
        # 如果有mask目录，加载mask
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
                mask = mask.astype(np.float32) / 255.0
                sample['mask'] = mask
        
        # 生成伪标签
        if self.use_pseudo_labels:
            result = self.decomposer.decompose(image)
            sample['pseudo_intrinsic'] = result['intrinsic']
            sample['pseudo_shading'] = result['shading']
            sample['pseudo_reflection'] = result['reflection']
            sample['pseudo_mask'] = result['highlight_mask']
        
        # 数据增强
        if self.transform:
            sample = self.transform(sample)
        
        # 转换为tensor
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1).float()
        
        if 'mask' in sample:
            sample['mask'] = torch.from_numpy(sample['mask']).unsqueeze(0).float()
        
        if self.use_pseudo_labels:
            sample['pseudo_intrinsic'] = torch.from_numpy(sample['pseudo_intrinsic']).permute(2, 0, 1).float()
            sample['pseudo_shading'] = torch.from_numpy(sample['pseudo_shading']).permute(2, 0, 1).float()
            sample['pseudo_reflection'] = torch.from_numpy(sample['pseudo_reflection']).permute(2, 0, 1).float()
            sample['pseudo_mask'] = torch.from_numpy(sample['pseudo_mask']).unsqueeze(0).float()
        
        return sample


class DecompositionTrainer:
    """
    分解网络训练器
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = DecompositionNet(
            input_channels=3,
            base_features=config.get('base_features', 64)
        ).to(self.device)
        
        # 损失函数
        self.criterion = DecompositionLoss(config)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config.get('log_dir', './logs'))
        
        # 检查点目录
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.start_epoch = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_losses = {}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, sample in enumerate(pbar):
            # 数据移到GPU
            image = sample['image'].to(self.device)
            
            # 前向传播
            intrinsic, shading, reflection, mask = self.model(image)
            
            # 计算损失
            losses = self.criterion(image, intrinsic, shading, reflection, mask)
            
            # 如果有伪标签，添加监督损失
            if 'pseudo_intrinsic' in sample:
                pseudo_intrinsic = sample['pseudo_intrinsic'].to(self.device)
                pseudo_shading = sample['pseudo_shading'].to(self.device)
                pseudo_reflection = sample['pseudo_reflection'].to(self.device)
                pseudo_mask = sample['pseudo_mask'].to(self.device)
                
                losses['pseudo_intrinsic'] = F.l1_loss(intrinsic, pseudo_intrinsic) * 0.1
                losses['pseudo_shading'] = F.l1_loss(shading, pseudo_shading) * 0.1
                losses['pseudo_reflection'] = F.l1_loss(reflection, pseudo_reflection) * 0.1
                losses['pseudo_mask'] = F.binary_cross_entropy(mask, pseudo_mask) * 0.1
                
                losses['total'] += (losses['pseudo_intrinsic'] + 
                                   losses['pseudo_shading'] + 
                                   losses['pseudo_reflection'] + 
                                   losses['pseudo_mask'])
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = []
                total_losses[k].append(v.item())
            
            # 更新进度条
            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items() if k in ['total', 'reconstruction']})
            
            # TensorBoard记录
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'Train/{k}', v.item(), global_step)
        
        # 返回平均损失
        avg_losses = {k: np.mean(v) for k, v in total_losses.items()}
        return avg_losses
    
    def validate(self, val_loader, epoch):
        """验证"""
        self.model.eval()
        total_losses = {}
        
        with torch.no_grad():
            for sample in tqdm(val_loader, desc='Validation'):
                image = sample['image'].to(self.device)
                
                # 前向传播
                intrinsic, shading, reflection, mask = self.model(image)
                
                # 计算损失
                losses = self.criterion(image, intrinsic, shading, reflection, mask)
                
                # 记录损失
                for k, v in losses.items():
                    if k not in total_losses:
                        total_losses[k] = []
                    total_losses[k].append(v.item())
        
        # 平均损失
        avg_losses = {k: np.mean(v) for k, v in total_losses.items()}
        
        # TensorBoard记录
        for k, v in avg_losses.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)
        
        return avg_losses
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # 保存最新的
        path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, path)
        
        # 保存最好的
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss: {loss:.6f}")
        
        # 定期保存
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """训练主循环"""
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*50)
            
            # 训练
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"Train - Total: {train_losses['total']:.4f}, Recon: {train_losses['reconstruction']:.4f}")
            
            # 验证
            val_losses = self.validate(val_loader, epoch)
            print(f"Val - Total: {val_losses['total']:.4f}, Recon: {val_losses['reconstruction']:.4f}")
            
            # 调整学习率
            self.scheduler.step(val_losses['total'])
            
            # 保存检查点
            is_best = val_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total']
            
            self.save_checkpoint(epoch, val_losses['total'], is_best)
            
            # 可视化结果
            if epoch % 5 == 0:
                self.visualize_results(val_loader, epoch)
    
    def visualize_results(self, val_loader, epoch):
        """可视化结果"""
        self.model.eval()
        
        with torch.no_grad():
            sample = next(iter(val_loader))
            image = sample['image'].to(self.device)
            
            # 预测
            intrinsic, shading, reflection, mask = self.model(image)
            
            # 选择第一张图像
            img = image[0].cpu()
            intr = intrinsic[0].cpu()
            shad = shading[0].cpu()
            refl = reflection[0].cpu()
            msk = mask[0].cpu()
            
            # 重构
            shad_3ch = shad.repeat(3, 1, 1)
            recon = intr * shad_3ch + refl
            
            # 创建网格图像
            grid = torch.cat([img, intr, shad_3ch, refl, msk.repeat(3, 1, 1), recon], dim=2)
            
            # 保存到TensorBoard
            self.writer.add_image('Decomposition', grid, epoch)


def main():
    parser = argparse.ArgumentParser(description='Train decomposition network')
    parser.add_argument('--config', type=str, default='train_config.yaml', help='Config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, help='Validation data directory')
    parser.add_argument('--mask_dir', type=str, help='Mask directory for weak supervision')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'base_features': 64,
        'lambda_recon': 1.0,
        'lambda_smooth': 0.1,
        'lambda_sparse': 0.01,
        'lambda_intrinsic': 0.1,
        'lambda_mask': 0.1,
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints'
    }
    
    # 加载配置文件
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # 创建数据集
    train_dataset = EndoscopeDecompositionDataset(
        image_dir=args.data_dir,
        mask_dir=args.mask_dir,
        image_size=(256, 320),
        use_pseudo_labels=True
    )
    
    val_dataset = None
    if args.val_dir:
        val_dataset = EndoscopeDecompositionDataset(
            image_dir=args.val_dir,
            mask_dir=None,
            image_size=(256, 320),
            use_pseudo_labels=False
        )
    else:
        # 使用训练集的一部分作为验证集
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = DecompositionTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(train_loader, val_loader, args.epochs)


if __name__ == '__main__':
    main()