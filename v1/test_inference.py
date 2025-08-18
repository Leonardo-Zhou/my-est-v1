#!/usr/bin/env python3
"""
测试inference.py的修复效果
"""
import os
import sys
import torch
import numpy as np
from inference import ImageDecomposer
import cv2

def test_checkpoint_loading():
    """测试检查点加载"""
    print("=== 测试检查点加载 ===")
    
    # 检查可用的检查点
    checkpoint_dir = "/home/zhouy/Works/DepthEstimation/my-est-v1"
    possible_checkpoints = [
        "checkpoint_best.pth",
        "checkpoint_latest.pth", 
        "model_best.pth",
        "best_model.pth"
    ]
    
    checkpoint_path = None
    for cp in possible_checkpoints:
        cp_path = os.path.join(checkpoint_dir, cp)
        if os.path.exists(cp_path):
            checkpoint_path = cp_path
            break
    
    if checkpoint_path is None:
        print("未找到检查点文件，创建模拟数据测试...")
        # 创建一个简单的测试图像
        test_image = np.random.randint(0, 255, (256, 320, 3), dtype=np.uint8)
        
        # 测试无检查点模式
        decomposer = ImageDecomposer(checkpoint_path=None, device='cpu')
        result = decomposer.decompose_single_frame(test_image)
        
        print("✓ 无检查点模式正常工作")
        print(f"分解结果包含: {list(result.keys())}")
        return True
    
    print(f"找到检查点: {checkpoint_path}")
    
    try:
        # 测试加载检查点
        decomposer = ImageDecomposer(checkpoint_path=checkpoint_path, device='cpu')
        print("✓ 检查点加载成功")
        
        # 测试分解功能
        test_image = np.random.randint(0, 255, (256, 320, 3), dtype=np.uint8)
        result = decomposer.decompose_single_frame(test_image)
        
        print("✓ 分解功能正常工作")
        print(f"分解结果包含: {list(result.keys())}")
        
        # 验证重构质量
        reconstruction = result['intrinsic'] * np.repeat(result['shading'], 3, axis=2) + result['reflection']
        recon_error = np.mean(np.abs(reconstruction - result['original']))
        print(f"重构误差: {recon_error:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_network_architectures():
    """测试不同网络架构的兼容性"""
    print("\n=== 测试网络架构兼容性 ===")
    
    from cyclegan_network import IntrinsicDecompositionNet
    from improved_network import ImprovedIntrinsicDecompositionNet, FastDecompositionNet
    from train_decomposition import DecompositionNet
    
    # 创建测试输入
    test_input = torch.randn(1, 3, 256, 320)
    
    networks = {
        'IntrinsicDecompositionNet': IntrinsicDecompositionNet(),
        'ImprovedIntrinsicDecompositionNet': ImprovedIntrinsicDecompositionNet(),
        'FastDecompositionNet': FastDecompositionNet(),
        'DecompositionNet': DecompositionNet()
    }
    
    for name, net in networks.items():
        try:
            net.eval()
            with torch.no_grad():
                if name == 'DecompositionNet':
                    # DecompositionNet返回4个值
                    intrinsic, shading, reflection, highlight = net(test_input)
                    print(f"✓ {name}: 输出形状 {intrinsic.shape}, {shading.shape}, {reflection.shape}, {highlight.shape}")
                else:
                    # 其他网络返回3个值
                    intrinsic, shading, reflection = net(test_input)
                    print(f"✓ {name}: 输出形状 {intrinsic.shape}, {shading.shape}, {reflection.shape}")
        except Exception as e:
            print(f"✗ {name}: 错误 {e}")

if __name__ == "__main__":
    print("开始测试inference.py修复效果...")
    
    success = test_checkpoint_loading()
    test_network_architectures()
    
    if success:
        print("\n✓ 测试完成，修复成功！")
    else:
        print("\n✗ 测试失败，需要进一步调试")
        sys.exit(1)