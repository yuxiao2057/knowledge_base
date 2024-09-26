from copy import deepcopy
import numpy as np

def nms(bounding_boxes, Nt):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)

    # 计算 n 个候选框的面积大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
    order = np.argsort(scores)

    picked_boxes = []  # 返回值
    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < Nt)
        order = order[left]
    return picked_boxes

def soft_nms(bboxes, Nt=0.3, sigma2=0.5, score_thresh=0.3, method=2):
    # 在 bboxes 之后添加对于的下标[0, 1, 2...], 最终 bboxes 的 shape 为 [n, 5], 前四个为坐标, 后一个为下标
    res_bboxes = deepcopy(bboxes)
    N = bboxes.shape[0]  # 总的 box 的数量
    indexes = np.array([np.arange(N)])  # 下标: 0, 1, 2, ..., n-1
    bboxes = np.concatenate((bboxes, indexes.T), axis=1)  # concatenate 之后, bboxes 的操作不会对外部变量产生影响
    # 计算每个 box 的面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # 找出 i 后面的最大 score 及其下标
        pos = i + 1
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        # 如果当前 i 的得分小于后面的最大 score, 则与之交换, 确保 i 上的 score 最大
        if scores[i] < maxscore:
            bboxes[[i, maxpos + i + 1]] = bboxes[[maxpos + i + 1, i]]
            scores[[i, maxpos + i + 1]] = scores[[maxpos + i + 1, i]]
            areas[[i, maxpos + i + 1]] = areas[[maxpos + i + 1, i]]
        # IoU calculate
        xx1 = np.maximum(bboxes[i, 0], bboxes[pos:, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[pos:, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[pos:, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[pos:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)
        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(iou.shape)
            weight[iou > Nt] = weight[iou > Nt] - iou[iou > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(iou * iou) / sigma2)
        else:  # original NMS
            weight = np.ones(iou.shape)
            weight[iou > Nt] = 0
        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    inds = bboxes[:, 5][scores > score_thresh]
    keep = inds.astype(int)
    return res_bboxes[keep]

def test_nms_no_boxes():
    # 测试没有候选框时的情况
    bounding_boxes = []
    Nt = 0.5
    expected_boxes = []
    picked_boxes = nms(bounding_boxes, Nt)
    assert np.array_equal(picked_boxes, expected_boxes)

def test_nms_single_box():
    # 测试只有一个候选框时的情况
    bounding_boxes = [[10, 10, 20, 20, 0.9]]
    Nt = 0.5
    expected_boxes = bounding_boxes
    picked_boxes = nms(bounding_boxes, Nt)
    assert np.array_equal(picked_boxes, expected_boxes)

def test_nms_multiple_boxes_no_overlap():
    # 测试多个没有重叠的候选框时的情况
    bounding_boxes = [
        [10, 10, 20, 20, 0.9],
        [30, 30, 40, 40, 0.8],
    ]
    Nt = 0.5
    expected_boxes = bounding_boxes
    picked_boxes = nms(bounding_boxes, Nt)
    assert np.array_equal(picked_boxes, expected_boxes)

def test_nms_multiple_boxes_with_overlap():
    # 测试多个有重叠的候选框时的情况
    bounding_boxes = [
        [10, 10, 20, 20, 0.9],
        [15, 15, 25, 25, 0.8],  # 与第一个框重叠
        [30, 30, 40, 40, 0.7],  # 与前两个不重叠
    ]
    Nt = 0.5
    # 期望的结果取决于NMS的实现，但在这个情况下，我们假设第一个框会被选中
    # （因为它有更高的置信度），并且第二个框（与第一个重叠）会被丢弃
    expected_boxes = [bounding_boxes[0], bounding_boxes[2]]
    picked_boxes = nms(bounding_boxes, Nt)
    # 注意：由于NMS可能返回不同但等价的顺序，所以我们不能直接比较数组
    # 我们应该比较每个框的坐标和分数
    assert len(picked_boxes) == len(expected_boxes)
    for pb, eb in zip(picked_boxes, expected_boxes):
        assert np.array_equal(pb[:4], eb[:4])  # 比较坐标
        assert np.isclose(pb[4], eb[4])  # 允许分数有一些小的差异（浮点数比较）


if __name__ == '__main__':
    test_nms_no_boxes()
    test_nms_single_box()
    test_nms_multiple_boxes_no_overlap()
    test_nms_multiple_boxes_with_overlap()