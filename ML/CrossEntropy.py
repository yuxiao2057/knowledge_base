import torch

def BinaryCrossEntropyLoss(y_pred, y_true):
    """
    Binary cross entropy loss function.
    """
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    bce_loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return bce_loss

def CrossEntropyLoss(y_pred, y_true):
    """
    Cross entropy loss function.
    """
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    ce_loss = -torch.mean(y_true * torch.log(y_pred))
    return ce_loss

def Softmax(y_pred):
    """
    Softmax function.
    """
    max_val = torch.max(y_pred, dim=-1, keepdim=True).values
    exp_scores = torch.exp(y_pred - max_val)
    
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
    return exp_scores / sum_exp
