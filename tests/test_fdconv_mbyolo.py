import torch
from ultralytics.nn.modules.mamba_yolo import RGBlock, SimpleStem, XSSBlock, VSSBlock

x = torch.randn(2, 128, 32, 32).cuda()

rg = RGBlock(in_features=128, hidden_features=256).cuda()
print('RGBlock ok:', rg(x).shape)

# stem = SimpleStem(3, 128).cuda()
# print('Stem ok:', stem(torch.randn(2, 3, 640, 640).cuda()).shape)

# xss = XSSBlock(in_channels=128, hidden_dim=256, n=1).cuda()
# print('XSSBlock ok:', xss(x).shape)

# vss = VSSBlock(in_channels=128, hidden_dim=256).cuda()
# print('VSSBlock ok:', vss(x).shape)