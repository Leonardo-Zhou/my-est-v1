import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from improved_network import ImprovedIntrinsicDecompositionNet, FastDecompositionNet


def test_improved_decomposition(image_path: str):
    """测试改进的分解效果"""
    
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 256))  # 调整到训练尺寸
    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    
    # 创建网络
    print("Testing Improved Network...")
    model = ImprovedIntrinsicDecompositionNet()
    model.eval()
    
    with torch.no_grad():
        intrinsic, shading, reflection = model(image_tensor)
    
    # 验证约束
    print("\n=== Constraint Verification ===")
    print(f"Intrinsic range: [{intrinsic.min():.3f}, {intrinsic.max():.3f}] (expected: [0.2, 0.8])")
    print(f"Shading range: [{shading.min():.3f}, {shading.max():.3f}] (expected: [0.3, 1.0])")
    print(f"Reflection range: [{reflection.min():.3f}, {reflection.max():.3f}] (expected: [0, 0.5])")
    
    # 检查重构
    shading_3ch = shading.repeat(1, 3, 1, 1)
    reconstruction = intrinsic * shading_3ch + reflection
    recon_error = torch.mean(torch.abs(reconstruction - image_tensor))
    print(f"Reconstruction error: {recon_error:.4f}")
    
    # 检查光照平滑性
    shading_np = shading[0, 0].numpy()
    grad_x = np.abs(np.diff(shading_np, axis=1))
    grad_y = np.abs(np.diff(shading_np, axis=0))
    smoothness = np.mean(grad_x) + np.mean(grad_y)
    print(f"Shading smoothness: {smoothness:.4f} (lower is smoother)")
    
    # 检查内在颜色亮度
    intrinsic_mean = torch.mean(intrinsic).item()
    print(f"Intrinsic mean brightness: {intrinsic_mean:.3f} (should be around 0.5)")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image / 255.0)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(intrinsic[0].permute(1, 2, 0).numpy())
    axes[0, 1].set_title(f'Intrinsic (mean: {intrinsic_mean:.2f})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(shading[0, 0].numpy(), cmap='gray')
    axes[0, 2].set_title(f'Shading (smooth: {smoothness:.3f})')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(reflection[0].permute(1, 2, 0).numpy())
    axes[1, 0].set_title('Reflection')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reconstruction[0].permute(1, 2, 0).numpy())
    axes[1, 1].set_title(f'Reconstruction (err: {recon_error:.3f})')
    axes[1, 1].axis('off')
    
    error_map = torch.abs(reconstruction[0] - image_tensor[0]).permute(1, 2, 0).numpy()
    axes[1, 2].imshow(error_map, cmap='jet')
    axes[1, 2].set_title('Error Map')
    axes[1, 2].axis('off')
    
    plt.suptitle('Improved Decomposition Test')
    plt.tight_layout()
    plt.savefig('improved_test.png', dpi=150)
    plt.show()
    
    return intrinsic, shading, reflection


def compare_networks(image_path: str):
    """比较改进网络和快速网络"""
    
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 256))
    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    
    # 测试改进网络
    model1 = ImprovedIntrinsicDecompositionNet()
    model1.eval()
    
    # 测试快速网络
    model2 = FastDecompositionNet()
    model2.eval()
    
    # 计算参数量
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    print(f"Improved Network parameters: {params1:,}")
    print(f"Fast Network parameters: {params2:,}")
    print(f"Reduction: {(1 - params2/params1)*100:.1f}%")
    
    # 测速
    import time
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model1(image_tensor)
            _ = model2(image_tensor)
    
    # 测试改进网络速度
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model1(image_tensor)
    time1 = time.time() - start
    
    # 测试快速网络速度
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model2(image_tensor)
    time2 = time.time() - start
    
    print(f"\nImproved Network: {time1:.2f}s for 100 iterations")
    print(f"Fast Network: {time2:.2f}s for 100 iterations")
    print(f"Speedup: {time1/time2:.2f}x")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input image')
    parser.add_argument('--compare', action='store_true', help='Compare networks')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_networks(args.input)
    else:
        test_improved_decomposition(args.input)