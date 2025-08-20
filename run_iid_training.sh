#!/bin/bash

# IID-based Depth Estimation Training Script
# Based on "Image Intrinsic-Based Unsupervised Monocular Depth Estimation in Endoscopy"

# Set your data path here
DATA_PATH="/path/to/your/data"
LOG_DIR="./logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Training with IID loss functions
python train_iid.py \
    --data_path $DATA_PATH \
    --log_dir $LOG_DIR \
    --model_name "iid_endoscopy_$(date +%Y%m%d_%H%M%S)" \
    --dataset endovis \
    --split endovis \
    --height 256 \
    --width 320 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 30 \
    --scheduler_step_size 10 \
    --lambda_ds 0.2 \
    --lambda_a 0.2 \
    --lambda_ms 1.0 \
    --lambda_es 0.01 \
    --num_layers 18 \
    --weights_init pretrained \
    --pose_model_input pairs \
    --log_frequency 200 \
    --save_frequency 1 \
    --num_workers 12

echo "Training completed!"
echo "Check tensorboard logs with: tensorboard --logdir=$LOG_DIR"