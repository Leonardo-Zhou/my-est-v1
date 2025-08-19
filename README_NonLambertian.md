# Non-Lambertian Image Intrinsic Decomposition for Endoscopic Depth Estimation

This project extends the original IID-SfmLearner to support **Non-Lambertian** image decomposition following the model `I = A × S + R` instead of the traditional Lambertian model `I = A × S`.

## Key Improvements

### Original Lambertian Model:
```
I = A × S
```
Where:
- `I`: Input image
- `A`: Albedo (reflectance)
- `S`: Shading

### New Non-Lambertian Model:
```
I = A × S + R
```
Where:
- `I`: Input image  
- `A`: Albedo (diffuse reflectance)
- `S`: Shading (illumination effects)
- `R`: Specular reflection (highlights)

## Architecture

The new model includes:

1. **Non-Lambertian Decompose Decoder**: Outputs three components (A, S, R) instead of two
2. **Enhanced Loss Functions**: 
   - Decomposition-Synthesis Loss: Ensures `I = A × S + R`
   - Albedo Consistency Loss: Albedo should be consistent across frames
   - Specular Smoothness Loss: Encourages sparse specular components
   - Standard depth and pose losses

## Usage

### Training

1. **Basic Training**:
```bash
python train_nonlambertian.py --data_path /mnt/data/publicData/MICCAI19_SCARED/train --log_dir ./logs
```

2. **Using the provided script**:
```bash
# Edit the DATA_PATH in the script first
bash run_nonlambertian_training.sh
```

### Testing

```bash
python test_nonlambertian.py \
    --load_weights_folder /path/to/trained/weights \
    --data_path /path/to/test/data
```

### Key Parameters

- `--reconstruction_constraint`: Weight for I = A*S + R constraint (default: 0.2)
- `--albedo_constraint`: Weight for albedo consistency across frames (default: 0.2)  
- `--reprojection_constraint`: Weight for final reconstruction loss (default: 1.0)
- `--specular_smoothness`: Weight for specular sparsity (default: 0.01)

## File Structure

### New Files:
- `networks/nonlambertian_decompose_decoder.py`: Three-component decoder
- `nonlambertian_trainer.py`: Training logic for I = A × S + R model
- `nonlambertian_options.py`: Command line options
- `train_nonlambertian.py`: Main training script
- `test_nonlambertian.py`: Testing and visualization script
- `run_nonlambertian_training.sh`: Training bash script

### Modified Files:
- `networks/__init__.py`: Added import for new decoder

## Expected Benefits

1. **Better Handling of Specular Reflections**: Explicitly models specular highlights common in endoscopic images
2. **Improved Decomposition Quality**: Separates diffuse and specular components
3. **Enhanced Depth Estimation**: Better image understanding leads to improved depth estimation
4. **Realistic Material Editing**: Enables separate control of albedo and specular properties

## Comparison with Original Model

The new model should provide:
- More accurate intrinsic decomposition for non-Lambertian surfaces
- Better performance on endoscopic images with specular reflections
- Improved depth estimation in challenging lighting conditions
- Physically plausible decomposition components

## Technical Details

### Network Architecture:
- **Shared Encoder**: ResNet-18 based feature extraction
- **Albedo Branch**: Full resolution decoder for material properties
- **Shading Branch**: Lower resolution decoder for smooth illumination
- **Specular Branch**: Full resolution decoder for sharp highlights

### Loss Function Weights:
- Total Loss = α₁×L_reconstruction + α₂×L_albedo + α₃×L_reprojection + α₄×L_specular + α₅×L_depth_smooth
- Recommended: α₁=0.2, α₂=0.2, α₃=1.0, α₄=0.01, α₅=0.01

## References

1. Original IID-SfmLearner paper: "Image Intrinsic-Based Unsupervised Monocular Depth Estimation in Endoscopy"
2. Non-Lambertian intrinsics: "Learning Non-Lambertian Object Intrinsics across ShapeNet Categories"

## Notes

- The model is backward compatible with the original codebase
- Both Lambertian and Non-Lambertian models can coexist
- Training data and evaluation protocols remain the same
- GPU memory requirements may be slightly higher due to the additional specular branch