import numpy as np

def nms(bboxes, threshold):
    """

    Args:
        bboxes (np.array): bboxes with shape (N, 5), 5 means (x1, y1, x2, y2, score)
        threshold (float): threshold for nms

    Returns:
        output_bboxes: output bboxes after nms
        
    """
    if len(bboxes) == 0:
        return []
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    sorted_idx = np.argsort(bboxes[:, 4])[::-1]
    output_bboxes = []
    while sorted_idx.size > 0:
        best = sorted_idx[0]
        output_bboxes.append(bboxes[best])
        
        x1_overlap = np.maximum(x1[best], x1[sorted_idx[1:]])
        y1_overlap = np.maximum(y1[best], y1[sorted_idx[1:]])
        x2_overlap = np.minimum(x2[best], x2[sorted_idx[1:]])
        y2_overlap = np.minimum(y2[best], y2[sorted_idx[1:]])
        
        w_overlap = np.maximum(0, x2_overlap - x1_overlap + 1)
        h_overlap = np.maximum(0, y2_overlap - y1_overlap + 1)
        area_overlap = w_overlap * h_overlap
        ious = area_overlap / (areas[sorted_idx[1:]] + areas[best] - area_overlap)
        
        left_boxes = np.where(ious < threshold)[0]
        sorted_idx = sorted_idx[left_boxes + 1]
    
    return output_bboxes

if __name__ == '__main__':
    np.random.seed(10)
    x1 = np.random.uniform(low = 0, high = 50, size= (100, 1))
    x2 = np.random.uniform(low = 150, high = 200, size= (100, 1))
    y1 = np.random.uniform(low = 0, high = 50, size= (100, 1))
    y2 = np.random.uniform(low = 150, high = 200, size= (100, 1))
    scores = np.random.rand(100)
    scores = scores[:, np.newaxis]
    bboxes = np.hstack([x1, y1, x2, y2, scores])
    output_bboxes = nms(bboxes, threshold=0.5)
    for bbox in output_bboxes:
        print(bbox.tolist())