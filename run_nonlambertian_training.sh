#!/bin/bash

# Non-Lambertian Image Intrinsic Decomposition Training Script
# This script trains the improved model that decomposes images as I = A × S + R
# instead of the original Lambertian model I = A × S

echo "Starting Non-Lambertian Model Training..."

# Set your data path here
DATA_PATH="./datasets/SCARED"  # Update this path
LOG_DIR="./logs"

# Training parameters
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_EPOCHS=30
HEIGHT=256
WIDTH=320

# Loss function weights for Non-Lambertian model
RECONSTRUCTION_CONSTRAINT=0.2      # Weight for I = A*S + R constraint
ALBEDO_CONSTRAINT=0.2             # Weight for albedo consistency across frames
REPROJECTION_CONSTRAINT=1.0       # Weight for final reconstruction loss
SPECULAR_SMOOTHNESS=0.01          # Weight for specular sparsity
DISPARITY_SMOOTHNESS=0.01         # Weight for depth smoothness

python train_nonlambertian.py \
    --data_path $DATA_PATH \
    --log_dir $LOG_DIR \
    --model_name "nonlambertian_model_$(date +%Y%m%d_%H%M%S)" \
    --split endovis \
    --dataset endovis \
    --height $HEIGHT \
    --width $WIDTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --reconstruction_constraint $RECONSTRUCTION_CONSTRAINT \
    --albedo_constraint $ALBEDO_CONSTRAINT \
    --reprojection_constraint $REPROJECTION_CONSTRAINT \
    --specular_smoothness $SPECULAR_SMOOTHNESS \
    --disparity_smoothness $DISPARITY_SMOOTHNESS \
    --num_workers 4 \
    --log_frequency 100 \
    --save_frequency 5

echo "Training completed!"

# Optional: Run testing after training
# Uncomment the following lines to test the trained model
# 
# echo "Running model testing..."
# python test_nonlambertian.py \
#     --load_weights_folder $LOG_DIR/nonlambertian_model_*/models/weights_* \
#     --data_path $DATA_PATH \
#     --split endovis \
#     --dataset endovis \
#     --height $HEIGHT \
#     --width $WIDTH
# 
# echo "Testing completed! Check ./nonlambertian_test_results/ for outputs."