import torch
import numpy as np
import cv2
import os
import argparse
from typing import Dict, Optional
import matplotlib.pyplot as plt
from improved_decomposition import PhysicallyBasedDecomposition, SimpleDecomposer


def test_decomposition(image_path: str, output_dir: str, use_simple: bool = False):
    """测试改进的分解方法"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image with shape: {image.shape}")
    
    # 选择分解器
    if use_simple:
        decomposer = SimpleDecomposer()
        print("Using simple decomposer for testing")
    else:
        decomposer = PhysicallyBasedDecomposition()
        print("Using physically-based decomposer")
    
    # 执行分解
    print("Performing decomposition...")
    result = decomposer.decompose(image)
    
    # 验证分解
    print("\n=== Decomposition Results ===")
    print(f"Intrinsic shape: {result['intrinsic'].shape}, range: [{result['intrinsic'].min():.3f}, {result['intrinsic'].max():.3f}]")
    print(f"Shading shape: {result['shading'].shape}, range: [{result['shading'].min():.3f}, {result['shading'].max():.3f}]")
    print(f"Reflection shape: {result['reflection'].shape}, range: [{result['reflection'].min():.3f}, {result['reflection'].max():.3f}]")
    print(f"Highlight mask shape: {result['highlight_mask'].shape}, range: [{result['highlight_mask'].min():.3f}, {result['highlight_mask'].max():.3f}]")
    
    # 检查高光mask是否有效
    highlight_pixels = np.sum(result['highlight_mask'] > 0.5)
    total_pixels = result['highlight_mask'].size
    highlight_percentage = (highlight_pixels / total_pixels) * 100
    print(f"Highlight pixels: {highlight_pixels}/{total_pixels} ({highlight_percentage:.2f}%)")
    
    # 检查反射是否非零
    reflection_energy = np.mean(result['reflection'])
    print(f"Reflection energy: {reflection_energy:.4f}")
    
    # 验证重构
    reconstruction_error = np.mean(np.abs(result['reconstruction'] - image/255.0))
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存各个分量
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_intrinsic.png"),
                cv2.cvtColor((result['intrinsic'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # 光照转为3通道保存
    shading_3ch = np.repeat(result['shading'], 3, axis=2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_shading.png"),
                (shading_3ch * 255).astype(np.uint8))
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_reflection.png"),
                cv2.cvtColor((result['reflection'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_highlight_mask.png"),
                (result['highlight_mask'] * 255).astype(np.uint8))
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_reconstruction.png"),
                cv2.cvtColor((result['reconstruction'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # 创建可视化
    visualize_decomposition_improved(result, image, save_path=os.path.join(output_dir, f"{base_name}_visualization.png"))
    
    print(f"\nResults saved to {output_dir}")
    
    return result


def visualize_decomposition_improved(result: Dict[str, np.ndarray], original_image: np.ndarray, save_path: Optional[str] = None):
    """改进的可视化函数"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 原始图像
    axes[0, 0].imshow(original_image if original_image.max() > 1 else original_image)
    axes[0, 0].set_title('Original Image I', fontsize=12)
    axes[0, 0].axis('off')
    
    # 内在颜色
    axes[0, 1].imshow(result['intrinsic'])
    axes[0, 1].set_title("Intrinsic Color I'", fontsize=12)
    axes[0, 1].axis('off')
    
    # 光照
    shading_vis = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
    axes[0, 2].imshow(shading_vis, cmap='gray' if result['shading'].shape[2] == 1 else None)
    axes[0, 2].set_title('Shading S', fontsize=12)
    axes[0, 2].axis('off')
    
    # 反射
    axes[0, 3].imshow(result['reflection'])
    axes[0, 3].set_title('Reflection R', fontsize=12)
    axes[0, 3].axis('off')
    
    # 高光mask
    axes[1, 0].imshow(result['highlight_mask'], cmap='hot')
    axes[1, 0].set_title('Highlight Mask', fontsize=12)
    axes[1, 0].axis('off')
    
    # I' * S
    intrinsic_shading = result['intrinsic'] * (np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading'])
    axes[1, 1].imshow(np.clip(intrinsic_shading, 0, 1))
    axes[1, 1].set_title("I' * S", fontsize=12)
    axes[1, 1].axis('off')
    
    # 重构
    axes[1, 2].imshow(np.clip(result['reconstruction'], 0, 1))
    axes[1, 2].set_title("Reconstruction I' * S + R", fontsize=12)
    axes[1, 2].axis('off')
    
    # 误差图
    original_norm = original_image / 255.0 if original_image.max() > 1 else original_image
    error = np.abs(result['reconstruction'] - original_norm)
    error_vis = error / error.max() if error.max() > 0 else error
    axes[1, 3].imshow(error_vis, cmap='jet')
    axes[1, 3].set_title(f'Error (mean: {np.mean(error):.4f})', fontsize=12)
    axes[1, 3].axis('off')
    
    plt.suptitle('Image Decomposition: I = I\' * S + R', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test improved image decomposition')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='./test_output', help='Output directory')
    parser.add_argument('--simple', action='store_true', help='Use simple decomposer for testing')
    
    args = parser.parse_args()
    
    # 测试分解
    test_decomposition(args.input, args.output, use_simple=args.simple)


if __name__ == '__main__':
    main()