import torch
import torch.nn as nn

class SPP(nn.Module):
    def __init__(self, num_levels):
        super(SPP, self).__init__()
        self.num_levels = num_levels  # 金字塔的层数
    
    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入特征图的维度
        output = []
        
        # 针对每个金字塔层，进行不同的池化操作
        for i in range(self.num_levels):
            level = 2 ** i  # 池化的核大小（例如 1x1，2x2，4x4 等）
            kernel_size = (h // level, w // level)
            stride = kernel_size
            pooling = nn.AdaptiveMaxPool2d(kernel_size)
            pooled = pooling(x)
            output.append(pooled.view(b, -1))  # 将池化后的特征展平并保存
            
        return torch.cat(output, dim=1)  # 将所有层的特征拼接在一起

# 测试SPP模块
spp = SPP(num_levels=3)
input_tensor = torch.randn(1, 256, 32, 32)  # 假设输入的特征图为 32x32 大小
output_tensor = spp(input_tensor)
print(output_tensor.shape)  # 输出大小： [1, 256 * (1^2 + 2^2 + 4^2)] = [1, 5376]