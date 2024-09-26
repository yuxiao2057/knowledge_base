import numpy as np

def softmax(X):
    
    exp_X = np.exp(X - np.max(X))

    return exp_X / np.sum(exp_X)

if __name__ == '__main__':
    X = np.random.uniform(0, 10, size=16)
    softmax_X = softmax(X)
    print(softmax_X)