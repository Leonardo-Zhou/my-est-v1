# 内窥镜图像分解系统

本项目实现了一个基于深度学习的内窥镜图像分解系统，将图像分解为 I = I' * S + R，其中：
- I': 内在颜色（物体本身的颜色）
- S: 光照分量
- R: 高光反射分量

## 主要特点

1. **高光检测**: 使用HSV和YCbCr空间的自适应阈值检测高光区域
2. **矩阵分解**: 使用SVT（奇异值阈值）和鲁棒PCA进行初始分解
3. **深度学习精炼**: 使用CycleGAN风格的网络进行无监督学习
4. **时间一致性**: 多帧处理时保持内在颜色的连续性
5. **后处理优化**: 迭代优化确保重构质量

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
python train.py --data_dir /path/to/endoscope/images --epochs 100 --batch_size 4
```

### 推理

处理单张图像：
```bash
python inference.py --input image.jpg --output ./output --type image --visualize
```

处理视频：
```bash
python inference.py --input video.mp4 --output ./output --type video --checkpoint checkpoints/best_model.pth
```

## 技术路线

### 步骤1: 预处理
- 在HSV或YCbCr空间检测潜在高光区域
- 使用亮度阈值，无需标签

### 步骤2: 初始分解
- 使用矩阵分解（SVT）粗分离R（高光）和临时I'*S

### 步骤3: 精炼
- 输入到CycleGAN风格的网络中
- 进行域转移（从有高光到无高光）
- 使用循环一致性和结构相似性损失

### 步骤4: 后处理
- 迭代优化S（通过光照平滑假设）
- 确保 I ≈ I' * S + R

## 文件结构

- `preprocessing.py`: 高光检测模块
- `matrix_decomposition.py`: SVT和RPCA分解
- `cyclegan_network.py`: 神经网络架构
- `temporal_consistency.py`: 时间一致性约束
- `postprocessing.py`: 后处理优化
- `train.py`: 训练脚本
- `inference.py`: 推理脚本
- `config.yaml`: 配置文件

## 配置参数

主要参数可在`config.yaml`中调整：
- `scale_factor`: 图像缩放因子（减小计算量）
- `sequence_length`: 时间一致性的序列长度
- `lambda_*`: 各种损失的权重

## 注意事项

- 输入图像建议大小为(1024, 1280, 3)，系统会自动缩放
- 训练时需要大量内窥镜图像数据
- GPU内存建议至少8GB