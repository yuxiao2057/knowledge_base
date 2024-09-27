import numpy as np

# 生成模拟数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 超参数
learning_rate = 0.1
n_iterations = 1000
m = len(X)

# 增加一列偏置项
X_b = np.hstack[np.ones((m, 1)), X]  # 在X前面添加一列1，作为偏置项

# 随机初始化权重参数
theta = np.random.randn(2, 1)

# 随机梯度下降
for iteration in range(n_iterations):
    # 每次随机抽取一个样本进行更新
    random_index = np.random.randint(m)
    xi = X_b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    
    # 计算预测值和误差
    gradients = 2 * xi.T @ (xi @ theta - yi)
    
    # 更新权重
    theta -= learning_rate * gradients

# 输出最终的theta值
print("最终的参数: \n", theta)
