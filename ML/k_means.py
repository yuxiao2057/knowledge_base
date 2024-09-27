import numpy as np

def kmeans_with_regularization(X, k, lambda_reg=0.1, max_iters=100, tol=1e-4):
    """
    带正则化的K-means算法，整合到一个单独的函数中。
    
    参数:
    X -- 数据集，形状为 (n_samples, n_features)
    k -- 簇的数量
    lambda_reg -- 正则化项的系数
    max_iters -- 最大迭代次数
    tol -- 收敛阈值
    
    返回:
    centroids -- 最终质心的位置
    labels -- 每个样本的簇标签
    """
    # 初始化质心
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    
    for i in range(max_iters):
        # 分配簇
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 更新质心
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            if np.any(labels == j):
                new_centroids[j] = X[labels == j].mean(axis=0)
            else:
                # 如果某个簇为空，重新随机选择一个质心
                new_centroids[j] = X[np.random.choice(X.shape[0])]
        
        # 计算损失（用于调试和收敛检查）
        loss = 0
        for j in range(k):
            cluster_points = X[labels == j]
            loss += np.sum((cluster_points - new_centroids[j]) ** 2)
        cluster_sizes = np.array([np.sum(labels == j) for j in range(k)])
        reg_term = lambda_reg * np.sum((cluster_sizes - np.mean(cluster_sizes)) ** 2)
        total_loss = loss + reg_term
        
        # 检查收敛
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
        print(f"Iteration {i+1}, loss: {total_loss}")
    
    return centroids, labels

if __name__ == "__main__":
    
    X = np.random.uniform(0, 100, size=(100, 5))
    centroids, labels = kmeans_with_regularization(X, 10)
    print(centroids)