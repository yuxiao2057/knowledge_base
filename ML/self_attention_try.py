import torch 
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    
    def __init__(self, input_dim, q_dim, v_dim):
        super().__init__()
        
        self.q_dim = q_dim
        self.k_dim = q_dim
        self.v_dim = v_dim
        
        self.linear_q = nn.Linear(input_dim, self.q_dim, bias=False)
        self.linear_k = nn.Linear(input_dim, self.k_dim, bias=False)
        self.linear_v = nn.Linear(input_dim, self.v_dim, bias=False)
        self.norm = math.sqrt(self.k_dim)
        
    def forward(self, x):
        
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)
        
        dist = torch.bmm(Q, K.transpose(1, 2)) / self.norm
        dist = dist - torch.max(dist, dim=-1, keepdim=True)[0]
        dist = torch.softmax(dist, dim=-1)
        
        attention = torch.bmm(dist, V)
        
        return attention
    
if __name__ == '__main__':
    
    batch_size = 5
    input_dim = 10
    seq_len = 8
    
    x = torch.rand(batch_size, seq_len, input_dim)
    self_attention = SelfAttention(input_dim, 10, 12)
    attention = self_attention(x)
    print(attention)