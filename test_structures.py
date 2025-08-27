import torch
import networks
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--structures', type=str, default='independent', 
                    choices=['independent', 'skip_links_0', 'skip_links_3', 'shared_encoder'])
parser.add_argument('--num_layers', type=int, default=18)
parser.add_argument('--weights_init', type=str, default='pretrained')

# 模拟opt对象
class Opt:
    def __init__(self):
        self.scales = range(4)
        self.num_layers = 18
        self.weights_init = 'pretrained'

opt = Opt()

# 测试不同的网络结构
def test_structures():
    print(f"Testing structure: {opt.structures}")
    
    # 创建示例输入
    input_image = torch.randn(1, 3, 256, 256)
    
    # 根据structures参数选择不同的网络结构
    if opt.structures == "shared_encoder":
        # 与深度网络共享编码器
        encoder = networks.ResnetEncoder(opt.num_layers, opt.weights_init == "pretrained")
    elif opt.structures == "skip_links_0":
        # 不使用跳跃连接
        encoder = networks.NonLambertianResnetEncoderV4(
            opt.num_layers, opt.weights_init == "pretrained", use_skips=False)
    elif opt.structures == "skip_links_3":
        # 仅在scale 3及更精细尺度使用跳跃连接
        encoder = networks.NonLambertianResnetEncoderV4(
            opt.num_layers, opt.weights_init == "pretrained")
    else:  # independent (default)
        # 独立编码器
        encoder = networks.NonLambertianResnetEncoderV4(
            opt.num_layers, opt.weights_init == "pretrained")
    
    # 将编码器移到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    input_image = input_image.to(device)
    
    # 前向传播
    features = encoder(input_image)
    print(f"Encoder output scales: {len(features)}")
    for i, f in enumerate(features):
        print(f"  Scale {i} feature shape: {f.shape}")
    
    # 解码器
    if opt.structures == "skip_links_0":
        # 不使用跳跃连接
        decoder = networks.NonLambertianDecomposeDecoderV4(
            encoder.num_ch_enc, opt.scales, use_skips=False)
    elif opt.structures == "skip_links_3":
        # 仅在scale 3及更精细尺度使用跳跃连接
        decoder = networks.NonLambertianDecomposeDecoderV4(
            encoder.num_ch_enc, opt.scales)
        # 设置解码器仅在精细尺度使用跳跃连接
        decoder.use_skips_fine_only = True
    else:  # independent, shared_encoder
        # 使用默认跳跃连接
        decoder = networks.NonLambertianDecomposeDecoderV4(
            encoder.num_ch_enc, opt.scales)
    
    # 将解码器移到GPU（如果可用）
    decoder.to(device)
    
    # 解码器前向传播
    outputs = decoder(features)
    print(f"Decoder outputs keys: {outputs.keys()}")
    
    print("Test passed!\n")

if __name__ == '__main__':
    # 解析参数
    args = parser.parse_args()
    opt.structures = args.structures
    opt.num_layers = args.num_layers
    opt.weights_init = args.weights_init
    
    # 测试所有结构
    structures = ['independent', 'skip_links_0', 'skip_links_3', 'shared_encoder']
    for structure in structures:
        opt.structures = structure
        try:
            test_structures()
        except Exception as e:
            print(f"Error testing structure {structure}: {e}\n")