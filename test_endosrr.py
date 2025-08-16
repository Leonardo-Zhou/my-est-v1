import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from typing import Dict, Optional

# 导入所有分解方法
from improved_decomposition import PhysicallyBasedDecomposition, SimpleDecomposer
from endosrr_decomposition import EndoSRRBasedDecomposition, ChromaticityBasedDecomposition


def compare_decomposition_methods(image_path: str, output_dir: str):
    """
    比较不同的分解方法
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image with shape: {image.shape}")
    
    # 准备所有分解方法
    methods = {
        'EndoSRR-Based': EndoSRRBasedDecomposition(use_iterative=True),
        'Physically-Based': PhysicallyBasedDecomposition(),
        'Chromaticity-Based': ChromaticityBasedDecomposition(),
        'Simple': SimpleDecomposer()
    }
    
    results = {}
    
    # 对每种方法进行测试
    for method_name, decomposer in methods.items():
        print(f"\n{'='*50}")
        print(f"Testing {method_name} method...")
        print('='*50)
        
        try:
            result = decomposer.decompose(image)
            results[method_name] = result
            
            # 验证结果
            print(f"\n{method_name} Results:")
            print(f"  Intrinsic range: [{result['intrinsic'].min():.3f}, {result['intrinsic'].max():.3f}]")
            print(f"  Shading range: [{result['shading'].min():.3f}, {result['shading'].max():.3f}]")
            print(f"  Reflection range: [{result['reflection'].min():.3f}, {result['reflection'].max():.3f}]")
            
            # 计算高光覆盖率
            highlight_coverage = np.mean(result['highlight_mask'] > 0.5) * 100
            print(f"  Highlight coverage: {highlight_coverage:.2f}%")
            
            # 计算重构误差
            original_norm = image / 255.0 if image.max() > 1 else image
            recon_error = np.mean(np.abs(result['reconstruction'] - original_norm))
            print(f"  Reconstruction error: {recon_error:.6f}")
            
            # 保存单独的结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            method_dir = os.path.join(output_dir, method_name.replace(' ', '_').lower())
            os.makedirs(method_dir, exist_ok=True)
            
            # 保存各个分量
            cv2.imwrite(
                os.path.join(method_dir, f"{base_name}_intrinsic.png"),
                cv2.cvtColor((result['intrinsic'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            
            shading_3ch = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
            cv2.imwrite(
                os.path.join(method_dir, f"{base_name}_shading.png"),
                (shading_3ch * 255).astype(np.uint8)
            )
            
            cv2.imwrite(
                os.path.join(method_dir, f"{base_name}_reflection.png"),
                cv2.cvtColor((result['reflection'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            
            cv2.imwrite(
                os.path.join(method_dir, f"{base_name}_mask.png"),
                (result['highlight_mask'] * 255).astype(np.uint8)
            )
            
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            continue
    
    # 创建比较图
    if results:
        create_comparison_figure(image, results, 
                               save_path=os.path.join(output_dir, 'comparison.png'))
    
    print(f"\nAll results saved to {output_dir}")
    
    return results


def create_comparison_figure(original_image: np.ndarray, 
                            results: Dict[str, Dict[str, np.ndarray]], 
                            save_path: Optional[str] = None):
    """
    创建比较不同方法的可视化图
    """
    num_methods = len(results)
    fig, axes = plt.subplots(num_methods + 1, 5, figsize=(20, 4 * (num_methods + 1)))
    
    # 显示原始图像
    original_norm = original_image / 255.0 if original_image.max() > 1 else original_image
    axes[0, 0].imshow(original_norm)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 隐藏第一行的其他格子
    for j in range(1, 5):
        axes[0, j].axis('off')
    
    # 显示每种方法的结果
    for i, (method_name, result) in enumerate(results.items(), 1):
        # Method name
        axes[i, 0].text(0.5, 0.5, method_name, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Intrinsic
        axes[i, 1].imshow(result['intrinsic'])
        axes[i, 1].set_title("Intrinsic I'", fontsize=10)
        axes[i, 1].axis('off')
        
        # Shading
        shading_vis = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
        axes[i, 2].imshow(shading_vis, cmap='gray' if result['shading'].shape[2] == 1 else None)
        axes[i, 2].set_title('Shading S', fontsize=10)
        axes[i, 2].axis('off')
        
        # Reflection
        axes[i, 3].imshow(result['reflection'])
        axes[i, 3].set_title('Reflection R', fontsize=10)
        axes[i, 3].axis('off')
        
        # Reconstruction error
        error = np.abs(result['reconstruction'] - original_norm)
        error_mean = np.mean(error)
        axes[i, 4].imshow(error, cmap='jet', vmin=0, vmax=0.1)
        axes[i, 4].set_title(f'Error (mean: {error_mean:.4f})', fontsize=10)
        axes[i, 4].axis('off')
    
    plt.suptitle('Comparison of Decomposition Methods\nI = I\' * S + R', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_single_method(image_path: str, output_dir: str, method: str = 'endosrr'):
    """
    测试单个方法
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 选择方法
    if method == 'endosrr':
        decomposer = EndoSRRBasedDecomposition(use_iterative=True)
    elif method == 'physical':
        decomposer = PhysicallyBasedDecomposition()
    elif method == 'chromaticity':
        decomposer = ChromaticityBasedDecomposition()
    else:
        decomposer = SimpleDecomposer()
    
    print(f"Using {method} decomposition method")
    
    # 执行分解
    result = decomposer.decompose(image)
    
    # 可视化
    visualize_endosrr_result(result, save_path=os.path.join(output_dir, 'visualization.png'))
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_intrinsic.png"),
        cv2.cvtColor((result['intrinsic'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    
    shading_3ch = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_shading.png"),
        (shading_3ch * 255).astype(np.uint8)
    )
    
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_reflection.png"),
        cv2.cvtColor((result['reflection'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    
    if 'image_no_reflection' in result:
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_no_reflection.png"),
            cv2.cvtColor((result['image_no_reflection'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
    
    print(f"Results saved to {output_dir}")
    
    return result


def visualize_endosrr_result(result: Dict[str, np.ndarray], save_path: Optional[str] = None):
    """
    可视化EndoSRR风格的结果
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 原始图像
    original = result['original']
    axes[0, 0].imshow(original if original.max() <= 1 else original/255)
    axes[0, 0].set_title('Original Image I', fontsize=12)
    axes[0, 0].axis('off')
    
    # 高光Mask
    axes[0, 1].imshow(result['highlight_mask'], cmap='hot')
    axes[0, 1].set_title('Specular Reflection Mask', fontsize=12)
    axes[0, 1].axis('off')
    
    # 反射分量
    axes[0, 2].imshow(result['reflection'])
    axes[0, 2].set_title('Reflection Component R', fontsize=12)
    axes[0, 2].axis('off')
    
    # 无反射图像（如果有）
    if 'image_no_reflection' in result:
        axes[0, 3].imshow(result['image_no_reflection'])
        axes[0, 3].set_title('Image without Reflection', fontsize=12)
    else:
        axes[0, 3].axis('off')
    axes[0, 3].axis('off')
    
    # 内在颜色
    axes[1, 0].imshow(result['intrinsic'])
    axes[1, 0].set_title("Intrinsic Albedo I'", fontsize=12)
    axes[1, 0].axis('off')
    
    # 光照
    shading_vis = np.repeat(result['shading'], 3, axis=2) if result['shading'].shape[2] == 1 else result['shading']
    axes[1, 1].imshow(shading_vis, cmap='gray')
    axes[1, 1].set_title('Shading S', fontsize=12)
    axes[1, 1].axis('off')
    
    # 重构
    axes[1, 2].imshow(np.clip(result['reconstruction'], 0, 1))
    axes[1, 2].set_title("Reconstruction I' * S + R", fontsize=12)
    axes[1, 2].axis('off')
    
    # 误差
    original_norm = original if original.max() <= 1 else original/255
    error = np.abs(result['reconstruction'] - original_norm)
    axes[1, 3].imshow(error, cmap='jet')
    axes[1, 3].set_title(f'Error (mean: {np.mean(error):.6f})', fontsize=12)
    axes[1, 3].axis('off')
    
    plt.suptitle('EndoSRR-Based Decomposition: I = I\' * S + R', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test EndoSRR-based decomposition methods')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='./endosrr_output', help='Output directory')
    parser.add_argument('--method', type=str, default='endosrr', 
                       choices=['endosrr', 'physical', 'chromaticity', 'simple', 'compare'],
                       help='Decomposition method to use')
    
    args = parser.parse_args()
    
    if args.method == 'compare':
        # 比较所有方法
        compare_decomposition_methods(args.input, args.output)
    else:
        # 测试单个方法
        test_single_method(args.input, args.output, args.method)


if __name__ == '__main__':
    main()