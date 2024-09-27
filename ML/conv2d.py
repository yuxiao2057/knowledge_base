import numpy as np 
def conv2d(img, in_channels, out_channels ,kernels, bias, stride=1, padding=0):
    N, C, H, W = img.shape 
    kh, kw = kernels.shape
    p = padding
    assert C == in_channels, "kernels' input channels do not match with img"
 
    if p:
        img = np.pad(img, ((0,0),(0,0),(p,p),(p,p)), 'constant') # padding along with all axis
 
    out_h = (H + 2*padding - kh) // stride + 1
    out_w = (W + 2*padding - kw) // stride + 1
 
    outputs = np.zeros([N, out_channels, out_h, out_w])
    # print(img)
    for n in range(N):
        for out in range(out_channels):
            for i in range(in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        for x in range(kh):
                            for y in range(kw):
                                outputs[n][out][h][w] += img[n][i][h * stride + x][w * stride + y] * kernels[x][y]
                if i == in_channels - 1:
                    outputs[n][out][:][:] += bias[n][out]
    return outputs