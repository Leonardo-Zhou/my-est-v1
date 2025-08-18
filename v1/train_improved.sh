#!/bin/bash
# 使用改进的网络重新训练

echo "Starting improved training..."

# 使用新的配置训练
python train.py \
    --data_dir /mnt/data/publicData/MICCAI19_SCARED/train/dataset7/keyframe4/image_02/data \
    --epochs 50 \
    --batch_size 4 \
    --config config.yaml

echo "Training completed!"

# 测试改进效果
echo "Testing improved model..."
python inference.py \
    --input /mnt/data/publicData/MICCAI19_SCARED/train/dataset7/keyframe4/image_02/data/0000000001.png \
    --output ./improved_results \
    --type image \
    --checkpoint ./checkpoints/checkpoint_epoch_50.pth \
    --visualize

echo "Done!"