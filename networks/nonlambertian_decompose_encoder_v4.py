import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # Use new torchvision API for loading pretrained weights
        if num_layers == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1
        elif num_layers == 50:
            weights = ResNet50_Weights.IMAGENET1K_V1
        
        # Load the pretrained model
        pretrained_model = models.resnet18(weights=weights) if num_layers == 18 else models.resnet50(weights=weights)
        
        # Copy the state dict and modify conv1 for multiple input images
        loaded = pretrained_model.state_dict()
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class NonLambertianResnetEncoderV4(nn.Module):
    """
    Non-Lambertian ResNet编码器 V4版本
    
    基于原版本的成功设计，为Non-Lambertian分解提供优化的特征提取
    保持与V4解码器的兼容性
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, use_skips=True):
        super(NonLambertianResnetEncoderV4, self).__init__()

        # 使用与原版本相同的通道配置
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.use_skips = use_skips

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            if pretrained:
                if num_layers == 18:
                    weights = ResNet18_Weights.IMAGENET1K_V1
                elif num_layers == 50:
                    weights = ResNet50_Weights.IMAGENET1K_V1
                else:
                    weights = None
                self.encoder = resnets[num_layers](weights=weights)
            else:
                self.encoder = resnets[num_layers](pretrained=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # 特征处理层 - 为不同的分解组件优化特征
        # 这些层将帮助编码器为albedo、shading和specular提供更好的特征表示
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_ch_enc[i], self.num_ch_enc[i], 1),
                nn.BatchNorm2d(self.num_ch_enc[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(self.num_ch_enc))
        ])
        
        # 为shading分支添加额外的平滑处理
        # 因为shading通常更平滑，我们为scale 2的特征添加额外的处理
        self.shading_feature_processor = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[2], self.num_ch_enc[2], 3, padding=1),
            nn.BatchNorm2d(self.num_ch_enc[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_ch_enc[2], self.num_ch_enc[2], 3, padding=1),
            nn.BatchNorm2d(self.num_ch_enc[2]),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        self.features = []
        processed_features = []
        
        # 第一层：conv1 + bn1 + relu
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        processed_features.append(self.feature_processors[0](x))
        
        # 第二层：maxpool + layer1
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        self.features.append(x)
        processed_features.append(self.feature_processors[1](x))
        
        # 第三层：layer2 (scale 2 - 对shading很重要)
        x = self.encoder.layer2(x)
        self.features.append(x)
        # 为shading分支添加额外的特征处理
        shading_features = self.shading_feature_processor(x)
        processed_features.append(self.feature_processors[2](shading_features))
        
        # 第四层：layer3
        x = self.encoder.layer3(x)
        self.features.append(x)
        processed_features.append(self.feature_processors[3](x))
        
        # 第五层：layer4 (最深层特征)
        x = self.encoder.layer4(x)
        self.features.append(x)
        processed_features.append(self.feature_processors[4](x))

        return processed_features