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

        # 确保每个头的维度一致
        assert q_dim % num_heads == 0, "q_dim must be divisible by num_heads"
        assert v_dim % num_heads == 0, "v_dim must be divisible by num_heads"

        # 每个头的 Q, K, V 维度
        self.head_dim_q = self.q_dim // num_heads
        self.head_dim_k = self.k_dim // num_heads
        self.head_dim_v = self.v_dim // num_heads

        # 定义线性变换用于生成 Q, K, V
        self.linear_q = nn.Linear(input_dim, q_dim, bias=False)
        self.linear_k = nn.Linear(input_dim, q_dim, bias=False)
        self.linear_v = nn.Linear(input_dim, v_dim, bias=False)

        # 输出线性层
        self.linear_out = nn.Linear(v_dim, v_dim, bias=False)

        # 归一化除数
        self.norm_devide = math.sqrt(self.head_dim_q)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)  # seq_len now in the second dimension

        # 计算 Q, K, V
        Q = self.linear_q(x)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, q_dim)
        K = self.linear_k(x)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, q_dim)
        V = self.linear_v(x)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, v_dim)

        # 分割为多个头 (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim_q).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim_q)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim_v).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim_v)

        # 计算注意力分数
        dist = torch.matmul(Q, K.transpose(-2, -1)) / self.norm_devide  # (batch_size, num_heads, seq_len, seq_len)
        dist = torch.softmax(dist, dim=-1)

        # 加权求和 V
        attention = torch.matmul(dist, V)  # (batch_size, num_heads, seq_len, head_dim_v)

        # 将多头拼接 (batch_size, seq_len, v_dim)
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # 输出通过线性层
        output = self.linear_out(attention)  # (batch_size, seq_len, v_dim)
        return output

if __name__ == '__main__':
    
    batch_size = 10
    input_dim = 10
    seq_len = 20
    num_heads = 4
    x = torch.randn(batch_size, seq_len, input_dim)  
    multi_head_attention = MultiHeadAttention(input_dim, 20, 40, num_heads)
    attention = multi_head_attention(x)
    print(attention.shape)  # 输出: (batch_size, seq_len, v_dim)
