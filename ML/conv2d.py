import numpy as np

def conv2d(img, in_channels, out_channels, kernels, bias, stride=1, padding=0):
    N, C, H, W = img.shape
    kh, kw = kernels.shape
    p = padding
    assert C == in_channels, "kernels' input channels do not match with img"
 
    if p:
        img = np.pad(img, ((0,0),(0,0),(p,p),(p,p)), 'constant')  # padding along all axes

    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1
    outputs = np.zeros([N, out_channels, out_h, out_w])

    for n in range(N):  # Batch size
        for out in range(out_channels):  # Output channels
            for i in range(in_channels):  # Input channels
                for h in range(out_h):  # Output height
                    for w in range(out_w):  # Output width
                        # 提取当前卷积窗口
                        img_patch = img[n, i, h * stride: h * stride + kh, w * stride: w * stride + kw]
                        # 使用矩阵逐元素乘法，然后求和
                        outputs[n, out, h, w] += np.sum(img_patch * kernels)
            # 加上偏置
            outputs[n, out, :, :] += bias[n][out]
    
    return outputs
