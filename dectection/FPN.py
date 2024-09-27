import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        self.backbone = backbone
        
        # 提取不同层的特征图，C3、C4、C5分别对应ResNet的不同层次
        self.layer1 = nn.Conv2d(1024, 256, kernel_size=1)  # C3  -> P3
        self.layer2 = nn.Conv2d(2048, 256, kernel_size=1)  # C4  -> P4
        self.layer3 = nn.Conv2d(512, 256, kernel_size=1)   # C5  -> P5
        
        # 在每层金字塔上进一步卷积操作
        self.p3_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.p4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.p5_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        # 使用ResNet提取C3、C4、C5层特征图
        c3 = self.backbone.layer3(x)  # 尺寸较大，语义较浅
        c4 = self.backbone.layer4(c3)  # 尺寸较小，语义较深
        c5 = self.backbone.layer5(c4)  # 尺寸最小，语义最深

        # 横向连接：通过1x1卷积降低通道数
        p5 = self.layer3(c5)
        p4 = self.layer2(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')  # 上采样并加上C4特征
        p3 = self.layer1(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')  # 上采样并加上C3特征

        # 每个P层进行3x3卷积操作
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        return [p3, p4, p5]  # 返回多尺度的特征图

# 创建ResNet backbone
resnet = resnet50(pretrained=True)
backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉ResNet的全连接层

# 实例化FPN
fpn = FPN(backbone=backbone)

# 测试FPN
input_tensor = torch.randn(1, 3, 256, 256)  # 假设输入图像为 256x256 大小
fpn_outputs = fpn(input_tensor)

# 输出多尺度特征图的尺寸
for i, f_map in enumerate(fpn_outputs):
    print(f"P{i+3} feature map shape: {f_map.shape}")
