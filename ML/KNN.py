import torch

def knn(x_train, y_train, x_test, k):
    """
    k-Nearest Neighbors classifier.
    Parameters:
        x_train (torch tensor): training data
        y_train (torch tensor): training labels
        x_test (torch tensor): test data
        k (int): number of nearest neighbors to consider
    Returns:
        predicted_labels (torch tensor): predicted labels for the test data
    """
    distances = torch.cdist(x_train, x_test)
    
    knn_indices = torch.topk(distances, k, dim = -1, largest=False).indices
    
    knn_labels = y_train[knn_indices]
    
    predicted_labels = []
    
    for labels in knn_labels:
        counts = torch.bincount(labels)
        predicted_labels.append(torch.argmax(counts))
    
    return torch.tensor(predicted_labels)