from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import networks
from layers import disp_to_depth
from utils import readlines
import datasets
from nonlambertian_options import NonLambertianOptions
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def test_nonlambertian_model():
    """Test the non-Lambertian decomposition model"""
    
    options = NonLambertianOptions()
    opts = options.parse()
    
    # Override some options for testing
    if not hasattr(opts, 'data_path'):
        opts.data_path = "./datasets"  # Set your data path
    if not hasattr(opts, 'load_weights_folder'):
        print("Please specify --load_weights_folder to load trained weights")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
    
    print("-> Loading model from ", opts.load_weights_folder)
    
    # Load models
    encoder_path = os.path.join(opts.load_weights_folder, "decompose_encoder.pth")
    decoder_path = os.path.join(opts.load_weights_folder, "decompose.pth")
    
    # Initialize networks
    encoder = networks.ResnetEncoder(opts.num_layers, False)
    decoder = networks.nonlambertian_decompose_decoder(encoder.num_ch_enc)
    
    # Load weights
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    loaded_dict_dec = torch.load(decoder_path, map_location=device)
    
    # Filter encoder weights
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    
    # Filter decoder weights  
    filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in decoder.state_dict()}
    decoder.load_state_dict(filtered_dict_dec)
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # Load test data
    datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
    dataset = datasets_dict[opts.dataset]
    
    fpath = os.path.join("splits", opts.split, "test_files.txt")
    if not os.path.exists(fpath):
        fpath = os.path.join("splits", opts.split, "val_files.txt")
    
    test_filenames = readlines(fpath)
    
    test_dataset = dataset(
        opts.data_path, test_filenames, opts.height, opts.width,
        [0], 4, is_train=False, img_ext='.png')
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, drop_last=False)
    
    print(f"-> Testing on {len(test_dataset)} images")
    
    # Create output directory
    output_dir = "./nonlambertian_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, inputs in enumerate(test_loader):
            if idx >= 10:  # Test on first 10 images
                break
                
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)
            
            input_color = inputs[("color", 0, 0)]
            
            # Get decomposition
            features = encoder(input_color)
            albedo, shading, specular = decoder(features)
            
            # Reconstruct image: I = A × S + R
            reconstructed = albedo * shading + specular
            
            # Convert to numpy for visualization
            input_np = input_color[0].cpu().numpy().transpose(1, 2, 0)
            albedo_np = albedo[0].cpu().numpy().transpose(1, 2, 0)
            shading_np = shading[0].cpu().numpy().transpose(1, 2, 0)
            specular_np = specular[0].cpu().numpy().transpose(1, 2, 0)
            reconstructed_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(input_np)
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(albedo_np)
            axes[0, 1].set_title('Albedo (A)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(shading_np, cmap='gray')
            axes[0, 2].set_title('Shading (S)')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(specular_np)
            axes[1, 0].set_title('Specular (R)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(reconstructed_np)
            axes[1, 1].set_title('Reconstructed (A×S + R)')
            axes[1, 1].axis('off')
            
            # Show difference
            diff = np.abs(input_np - reconstructed_np)
            axes[1, 2].imshow(diff)
            axes[1, 2].set_title('Reconstruction Error')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'decomposition_{idx:03d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save individual components
            cv2.imwrite(os.path.join(output_dir, f'input_{idx:03d}.png'), 
                       (input_np * 255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(output_dir, f'albedo_{idx:03d}.png'), 
                       (albedo_np * 255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(output_dir, f'shading_{idx:03d}.png'), 
                       (shading_np * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_dir, f'specular_{idx:03d}.png'), 
                       (specular_np * 255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(output_dir, f'reconstructed_{idx:03d}.png'), 
                       (reconstructed_np * 255).astype(np.uint8)[:,:,::-1])
            
            print(f"Processed image {idx+1}/10")
    
    print(f"-> Results saved to {output_dir}")

def compare_lambertian_vs_nonlambertian():
    """Compare original Lambertian model (I = A × S) vs Non-Lambertian model (I = A × S + R)"""
    
    options = NonLambertianOptions()
    opts = options.parse()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
    
    # Load both models for comparison
    print("-> Loading models for comparison...")
    
    # You would need to specify paths to both models
    lambertian_weights = opts.load_weights_folder + "_lambertian"  # Original model
    nonlambertian_weights = opts.load_weights_folder + "_nonlambertian"  # New model
    
    print("-> Comparison functionality would compare:")
    print("   - Reconstruction quality (PSNR, SSIM)")
    print("   - Depth estimation accuracy") 
    print("   - Component plausibility (albedo smoothness, specular sparsity)")
    print("   - Runtime performance")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Non-Lambertian Model")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                       help="Path to the folder containing model weights")
    parser.add_argument("--data_path", type=str, default="./datasets",
                       help="Path to test data")
    parser.add_argument("--split", type=str, default="endovis", 
                       choices=["endovis", "hamlyn"])
    parser.add_argument("--dataset", type=str, default="endovis",
                       choices=["endovis", "hamlyn"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--compare", action="store_true", 
                       help="Run comparison between Lambertian and Non-Lambertian models")
    
    args = parser.parse_args()
    
    # Convert args to opts format expected by the code
    class OptsClass:
        pass
    
    opts = OptsClass()
    for key, value in vars(args).items():
        setattr(opts, key, value)
    
    if args.compare:
        compare_lambertian_vs_nonlambertian()
    else:
        test_nonlambertian_model()