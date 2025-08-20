from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import networks
from layers import disp_to_depth
from utils import readlines
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test IID-based depth estimation')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the data')
    parser.add_argument('--load_weights_folder', type=str, required=True,
                        help='path to the trained model')
    parser.add_argument('--dataset', type=str, default='endovis',
                        choices=['endovis', 'hamlyn'],
                        help='dataset to test on')
    parser.add_argument('--split', type=str, default='endovis',
                        choices=['endovis', 'hamlyn'],
                        help='which split to use')
    parser.add_argument('--height', type=int, default=256,
                        help='input image height')
    parser.add_argument('--width', type=int, default=320,
                        help='input image width')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='minimum depth')
    parser.add_argument('--max_depth', type=float, default=150.0,
                        help='maximum depth')
    parser.add_argument('--save_pred_disps', action='store_true',
                        help='save predicted disparities')
    parser.add_argument('--output_dir', type=str, default='iid_test_results',
                        help='output directory for results')
    parser.add_argument('--num_layers', type=int, default=18,
                        choices=[18, 34, 50, 101, 152],
                        help='number of resnet layers')
    parser.add_argument('--scales', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='scales used in the loss')
    parser.add_argument('--frame_ids', nargs='+', type=int, default=[0, -1, 1],
                        help='frames to load')
    
    return parser.parse_args()

def test():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load models
    models = {}
    
    # Encoder
    models["encoder"] = networks.ResnetEncoder(args.num_layers, False)
    models["encoder"].to(device)
    
    # Depth decoder
    models["depth"] = networks.DepthDecoder(models["encoder"].num_ch_enc, args.scales)
    models["depth"].to(device)
    
    # Decomposition encoder and decoder
    models["decompose_encoder"] = networks.ResnetEncoder(args.num_layers, False)
    models["decompose_encoder"].to(device)
    
    models["decompose"] = networks.decompose_decoder(models["decompose_encoder"].num_ch_enc, args.scales)
    models["decompose"].to(device)
    
    # Load weights
    for model_name in models:
        path = os.path.join(args.load_weights_folder, "{}.pth".format(model_name))
        if os.path.exists(path):
            model_dict = models[model_name].state_dict()
            pretrained_dict = torch.load(path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[model_name].load_state_dict(model_dict)
            print(f"Loaded {model_name} weights")
        else:
            print(f"Warning: {model_name} weights not found at {path}")
    
    # Set to eval mode
    for model in models.values():
        model.eval()
    
    # Load dataset
    datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
    dataset_class = datasets_dict[args.dataset]
    
    fpath = os.path.join(os.path.dirname(__file__), "splits", args.split, "{}_files.txt")
    test_filenames = readlines(fpath.format("test"))
    img_ext = '.png'
    
    test_dataset = dataset_class(
        args.data_path, test_filenames, args.height, args.width,
        args.frame_ids, 4, is_train=False, img_ext=img_ext)
    
    test_loader = DataLoader(
        test_dataset, 1, False, num_workers=1, pin_memory=True, drop_last=False)
    
    print(f"Testing on {len(test_dataset)} images")
    
    # Test loop
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)
            
            # Depth estimation
            features = models["encoder"](inputs["color_aug", 0, 0])
            outputs = models["depth"](features)
            
            # Intrinsic decomposition
            decompose_features = models["decompose_encoder"](inputs["color_aug", 0, 0])
            albedo, shading = models["decompose"](decompose_features)
            
            # Get depth
            disp = outputs[("disp", 0)]
            _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
            
            # Convert to numpy
            depth_np = depth[0, 0].cpu().numpy()
            albedo_np = albedo[0].cpu().numpy().transpose(1, 2, 0)
            shading_np = shading[0].cpu().numpy().transpose(1, 2, 0)
            input_np = inputs[("color", 0, 0)][0].cpu().numpy().transpose(1, 2, 0)
            
            # Reconstruct image
            reconstructed_np = albedo_np * shading_np
            
            # Save results
            if args.save_pred_disps:
                # Normalize depth for visualization
                depth_vis = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                
                # Save images
                cv2.imwrite(os.path.join(args.output_dir, f"input_{i:03d}.png"), 
                           (input_np * 255).astype(np.uint8)[:, :, ::-1])
                cv2.imwrite(os.path.join(args.output_dir, f"depth_{i:03d}.png"), depth_vis)
                cv2.imwrite(os.path.join(args.output_dir, f"albedo_{i:03d}.png"), 
                           (albedo_np * 255).astype(np.uint8)[:, :, ::-1])
                cv2.imwrite(os.path.join(args.output_dir, f"shading_{i:03d}.png"), 
                           (shading_np * 255).astype(np.uint8)[:, :, ::-1])
                cv2.imwrite(os.path.join(args.output_dir, f"reconstructed_{i:03d}.png"), 
                           (reconstructed_np * 255).astype(np.uint8)[:, :, ::-1])
                
                # Create decomposition visualization
                decomposition_vis = np.hstack([
                    input_np, albedo_np, shading_np, reconstructed_np
                ])
                cv2.imwrite(os.path.join(args.output_dir, f"decomposition_{i:03d}.png"), 
                           (decomposition_vis * 255).astype(np.uint8)[:, :, ::-1])
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_dataset)} images")
    
    print(f"Testing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    test()