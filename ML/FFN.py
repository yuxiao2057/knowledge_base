import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义一个简单的前向传播神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        # 前向传播逻辑
        x = F.relu(self.fc1(x))  # 隐藏层使用 ReLU 激活函数
        x = self.fc2(x)  # 输出层 (线性层)
        return x

# 设置超参数
input_size = 3   # 输入层维度
hidden_size = 5  # 隐藏层神经元数量
output_size = 2  # 输出层维度
learning_rate = 0.01

# 创建模型
model = SimpleNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建一些示例输入数据 (batch_size x input_size)
inputs = torch.randn(10, input_size)  # 10 个样本
targets = torch.randn(10, output_size)  # 对应的目标值

# 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # 打印损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed.")
