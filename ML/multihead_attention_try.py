import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, q_dim, v_dim, num_heads):
        super().__init__()
        
        self.input_dim = input_dim
        self.q_dim = q_dim
        self.k_dim = q_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        
        assert q_dim % num_heads == 0, "Dimension of Q and K must be devisable by number fo heads"
        assert v_dim % num_heads == 0, "Dimension of V must be devisable by number fo heads"
        
        self.head_q_dim = self.q_dim // self.num_heads
        self.head_k_dim = self.k_dim // self.num_heads
        self.head_v_dim = self.v_dim // self.num_heads
        
        self.linear_q = nn.Linear(self.input_dim, self.q_dim, bias=False)
        self.linear_k = nn.Linear(self.input_dim, self.k_dim, bias=False)
        self.linear_v = nn.Linear(self.input_dim, self.v_dim, bias=False)
        
        self.linear_output = nn.Linear(self.v_dim, self.v_dim, bias=False)
        
        self.devide_norm = math.sqrt(self.k_dim)
        
    def forward(self, X):
        
        # X:(B, seq_len, input_dim)
        
        batch_size = X.size(0)
        seq_len = X.size(1)
        
        Q = self.linear_q(X)
        K = self.linear_k(X)
        V = self.linear_v(X)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_q_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_k_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_v_dim).transpose(1, 2)
        
        dist = torch.matmul(Q, K.transpose(-2, -1)) / self.devide_norm
        dist = torch.softmax(dist, dim=-1)
        
        attention = torch.matmul(dist, V).transpose(1, 2)
        attention = attention.reshape(batch_size, seq_len, -1)
        
        output = self.linear_output(attention)
        
        return output
    
if __name__ == '__main__':
    
    batch_size = 20
    seq_len = 20
    input_dim = 50
    num_heads = 5
    
    input_seq = torch.rand(size=(batch_size, seq_len, input_dim))
    print(f"input shape: {input_seq.shape}")
    
    multi_head_attention = MultiHeadAttention(input_dim, 20, 30, num_heads)
    output_seq = multi_head_attention(input_seq)
    
    print(f"output shape: {output_seq.shape}")