#!/bin/bash

# 内存优化的V4训练脚本
# 设置CUDA内存管理环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 运行训练，使用更小的batch size和内存优化设置
python train_nonlambertian_v4.py \
    --model_name nonlambertian_v4_memory_optimized \
    --batch_size 4 \
    --height 192 \
    --width 640 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --log_frequency 500 \
    --save_frequency 2 \
    --reconstruction_constraint 1.0 \
    --albedo_constraint 0.05 \
    --specular_smoothness 0.005 \
    --specular_l1_sparsity 0.005 \
    --shading_smoothness 0.001 \
    --albedo_smoothness 0.001 \
    --progressive_specular_weight \
    --specular_warmup_epochs 10 \
    --use_shading_scale2_start