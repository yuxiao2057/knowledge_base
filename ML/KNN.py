import numpy as np

class KNN:
    def __init__(self, k, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # 计算距离
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # 选取最近的k个样本的标签
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        # 返回最常见的标签
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]

np.random.seed(42)  # 固定随机种子以保证结果可复现
X_train = np.random.rand(1000, 5)  # 1000个训练样本，每个样本5个特征
y_train = np.random.randint(0, 5, 1000)  # 1000个标签，标签值为0, 1, 2, 3, 4中的一个
X_test = np.random.rand(10, 5)  # 10个测试样本

# 测试KNN
knn = KNN(k=5, X_train=X_train, y_train=y_train)
predictions = knn.predict(X_test)

print("Test sets:\n", X_test)
print("Predicted labels for the test set:\n", predictions)