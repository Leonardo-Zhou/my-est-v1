# IID-based Depth Estimation (Adapted to Your Project)

This implementation applies the loss functions from the paper "Image Intrinsic-Based Unsupervised Monocular Depth Estimation in Endoscopy" to your existing project, supporting both your standard model and non-Lambertian model.

## Overview

The IID (Intrinsic Image Decomposition) approach is adapted to work with your project's existing architecture:
- **Standard Model**: `I = reflectance × light` (from your `trainer.py`)
- **Non-Lambertian Model**: `I = albedo × shading + specular` (from your `nonlambertian_trainer.py`)

This handles illumination variations that violate photometric consistency assumptions in traditional depth estimation methods.

## Key Components

### Loss Functions

The implementation includes five main loss functions adapted to your project:

1. **Decomposing-Synthesis Loss (Lds)**: Ensures faithful reconstruction
   - Standard: `I = reflectance × light`
   - Non-Lambertian: `I = albedo × shading + specular`
2. **Component Consistency Loss (La)**: Enforces consistency across frames
   - Standard: reflectance consistency
   - Non-Lambertian: albedo consistency  
3. **Mapping-Synthesis Loss (Lms)**: Final reconstruction with illumination adjustment
4. **Edge-Aware Depth Smooth Loss (Les)**: Depth smoothness with edge preservation
5. **Specular Smoothness Loss**: Encourages sparsity in specular component (non-Lambertian only)

### Network Architecture

- **Intrinsic Decomposition Module**: Decomposes images into albedo and shading
- **Synthesis Reconstruction Module**: Reconstructs images with illumination adjustment
- **Depth Estimation Network**: Standard depth estimation backbone
- **Pose Estimation Network**: Camera motion estimation

## Usage

### Training

1. **Basic training with default parameters:**
```bash
python train_iid.py --data_path /path/to/data --log_dir ./logs
```

2. **Training with custom loss weights:**
```bash
python train_iid.py \
    --data_path /path/to/data \
    --log_dir ./logs \
    --lambda_ds 0.2 \
    --lambda_a 0.2 \
    --lambda_ms 1.0 \
    --lambda_es 0.01
```

3. **Using the provided script:**
```bash
# Edit DATA_PATH in run_iid_training.sh first
chmod +x run_iid_training.sh
./run_iid_training.sh
```

### Testing

```bash
python test_iid.py \
    --data_path /path/to/data \
    --load_weights_folder /path/to/trained/model \
    --save_pred_disps \
    --output_dir iid_results
```

## Loss Function Details

### Mathematical Formulation

The total loss is computed as:
```
L_total = λ_ds * L_ds + λ_a * L_a + λ_ms * L_ms + λ_es * L_es
```

Where:
- **L_ds**: Decomposing-Synthesis Loss using SSIM + L1
- **L_a**: Albedo consistency loss (L1 distance)
- **L_ms**: Mapping-Synthesis Loss using SSIM + L1  
- **L_es**: Edge-aware depth smoothness loss

### Default Weights (adapted to your project)

- `λ_ds = 0.2` (Decomposing-Synthesis)
- `λ_a = 0.2` (Component Consistency - reflectance/albedo)
- `λ_ms = 1.0` (Mapping-Synthesis)
- `λ_es = 0.01` (Edge-aware Smooth)
- `λ_spec = 0.01` (Specular Smoothness - non-Lambertian only)

## Key Differences from Original Implementation

### Advantages of IID Approach:

1. **Illumination Robustness**: Handles lighting variations in endoscopic images
2. **No Photometric Consistency Assumption**: Works when traditional assumptions fail
3. **Intrinsic Component Supervision**: Additional supervision from albedo/shading decomposition
4. **End-to-End Training**: Joint optimization of all components

### Compared to Your Original Implementation:

- **Original**: Uses reflectance consistency and reconstruction constraints
- **IID**: Uses intrinsic decomposition with albedo consistency and mapping-synthesis losses
- **Benefit**: Better handling of specular reflections and illumination changes

## File Structure

```
├── iid_trainer.py          # Main trainer with IID loss functions
├── iid_options.py          # Command line options for IID training
├── train_iid.py           # Training script
├── test_iid.py            # Testing script with visualization
├── run_iid_training.sh    # Convenient training script
└── README_IID.md          # This file
```

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir=./logs
```

The implementation logs:
- Individual loss components (Lds, La, Lms, Les)
- Total loss
- Depth maps
- Intrinsic components (albedo, shading)
- Reconstructed images

## Expected Results

Based on the paper, you should expect:
- Better depth estimation in regions with illumination changes
- More consistent albedo maps across frames
- Improved handling of specular reflections
- Superior performance compared to traditional photometric consistency methods

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Loss not decreasing**: Check learning rate and loss weights
3. **Poor decomposition**: Ensure proper data preprocessing and augmentation
4. **Model not loading**: Verify model paths and architecture consistency

## Citation

If you use this implementation, please cite the original paper:
```
@article{li2024image,
  title={Image Intrinsic-Based Unsupervised Monocular Depth Estimation in Endoscopy},
  author={Li, Bojian and Liu, Bo and Zhu, Miao and Luo, Xiaoyan and Zhou, Fugen},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024}
}
```