import numpy as np

def k_means_with_reg(X, k, lambda_reg = 0.1, max_iter = 100, res = 1e-4):
    
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices];
    
    for i in range(max_iter):
        
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros((k, X.shape[1]))
        
        for j in range(k):
            if np.any(labels == j):
                new_centroids = np.mean(X[labels == j], axis=0)
            else:
                new_centroids = X[np.random.choice(X.shape[0])]
        
        loss = 0
        reg = 0
        clusters = 0
        
        for j in range(k):
            loss += np.sum((X[labels == j] - new_centroids[j]) ** 2)
            
        clusters = np.array((X[labels == j] for j in range(k)))
        reg = lambda_reg * np.sum((clusters - np.mean(clusters)) ** 2)
        
        total_loss = loss + reg
        
        if (np.norm(new_centroids - centroids) < res):
            break
            
        centroids = new_centroids
        print(f"Iterations:{i + 1}, Loss:{total_loss}")
        
    return centroids, labels