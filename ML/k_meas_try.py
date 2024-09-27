import numpy as np

def k_means_with_reg(X, k, lambda_reg = 0.1, max_iter = 100, eps = 1e-4):
    
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    
    for iteration in range(max_iter):
        
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            if np.any(labels == j):
                # 计算新的质心，同时加入 L2 正则化
                new_centroids[j] = (X[labels == j].mean(axis=0) + lambda_reg * centroids[j]) / (1 + lambda_reg)
            else:
                new_centroids[j] = X[np.random.choice(X.shape[0])]
                
        loss = 0
        
        for j in range(k):
            loss += np.sum((X[labels == j] - new_centroids[j]) ** 2)
           
        cluster_sizes = np.array([np.sum(labels == j) for j in range(k)])
        reg = lambda_reg * np.sum((cluster_sizes - np.mean(cluster_sizes)) ** 2)
        
        total_loss = loss + reg
        
        if (np.linalg.norm(new_centroids - centroids) < eps):
            return new_centroids, labels
        
        centroids = new_centroids
        print(f"Iteration:{iteration}, Loss:{total_loss}")
        
    return centroids, labels
        
if __name__ == "__main__":
    
    X = np.random.uniform(0, 100, size=(1000, 5))
    centroids, labels = k_means_with_reg(X, 10)
    print(centroids)
    print(labels)