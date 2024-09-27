import numpy as np
import torch

def softmax(X):
    
    exp_X = torch.exp(X - torch.max(X))

    return exp_X / torch.sum(exp_X)

if __name__ == '__main__':
    X = torch.rand(16)*10
    softmax_X = softmax(X)
    print(softmax_X)