import torch
import math

def iou_loss(pred, target, reduction='mean', eps=1e-6):
    """
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction: "mean" or "sum"
    return: loss
    """
    # 求pred, target面积
    pred_widths = (pred[:, 2] - pred[:, 0] + 1.).clamp(0)
    pred_heights = (pred[:, 3] - pred[:, 1] + 1.).clamp(0)
    target_widths = (target[:, 2] - target[:, 0] + 1.).clamp(0)
    target_heights = (target[:, 3] - target[:, 1] + 1.).clamp(0)
    pred_areas = pred_widths * pred_heights
    target_areas = target_widths * target_heights

    # 求pred, target相交面积
    inter_xmins = torch.maximum(pred[:, 0], target[:, 0])
    inter_ymins = torch.maximum(pred[:, 1], target[:, 1])
    inter_xmaxs = torch.minimum(pred[:, 2], target[:, 2])
    inter_ymaxs = torch.minimum(pred[:, 3], target[:, 3])
    inter_widths = torch.clamp(inter_xmaxs - inter_xmins + 1.0, min=0.)
    inter_heights = torch.clamp(inter_ymaxs - inter_ymins + 1.0, min=0.)
    inter_areas = inter_widths * inter_heights

    # 求iou
    ious = torch.clamp(inter_areas / (pred_areas + target_areas - inter_areas), min=eps)
    if reduction == 'mean':
        loss = torch.mean(-torch.log(ious))
    elif reduction == 'sum':
        loss = torch.sum(-torch.log(ious))
    else:
        raise NotImplementedError

    return loss

def giou_loss(pred, target, reduction='mean', eps=1e-6):
    """
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction: "mean" or "sum"
    return: loss
    """
    # 求pred, target面积
    pred_widths = (pred[:, 2] - pred[:, 0] + 1.).clamp(0)
    pred_heights = (pred[:, 3] - pred[:, 1] + 1.).clamp(0)
    target_widths = (target[:, 2] - target[:, 0] + 1.).clamp(0)
    target_heights = (target[:, 3] - target[:, 1] + 1.).clamp(0)
    pred_areas = pred_widths * pred_heights
    target_areas = target_widths * target_heights

    # 求pred, target相交面积
    inter_xmins = torch.maximum(pred[:, 0], target[:, 0])
    inter_ymins = torch.maximum(pred[:, 1], target[:, 1])
    inter_xmaxs = torch.minimum(pred[:, 2], target[:, 2])
    inter_ymaxs = torch.minimum(pred[:, 3], target[:, 3])
    inter_widths = torch.clamp(inter_xmaxs - inter_xmins + 1.0, min=0.)
    inter_heights = torch.clamp(inter_ymaxs - inter_ymins + 1.0, min=0.)
    inter_areas = inter_widths * inter_heights

    # 求iou
    unions = pred_areas + target_areas - inter_areas
    ious = torch.clamp(inter_areas / unions, min=eps)

    # 求最小外接矩形
    outer_xmins = torch.minimum(pred[:, 0], target[:, 0])
    outer_ymins = torch.minimum(pred[:, 1], target[:, 1])
    outer_xmaxs = torch.maximum(pred[:, 2], target[:, 2])
    outer_ymaxs = torch.maximum(pred[:, 3], target[:, 3])
    outer_widths = (outer_xmaxs - outer_xmins + 1).clamp(0.)
    outer_heights = (outer_ymaxs - outer_ymins + 1).clamp(0.)
    outer_areas = outer_heights * outer_widths

    gious = ious - (outer_areas - unions) / outer_areas
    gious = gious.clamp(min=-1.0, max=1.0)
    if reduction == 'mean':
        loss = torch.mean(1 - gious)
    elif reduction == 'sum':
        loss = torch.sum(1 - gious)
    else:
        raise NotImplementedError
    return loss


def diou_loss(pred, target, reduce='mean', eps=1e-6):
    """
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction: "mean" or "sum"
    return: loss
    """
    # 求pred, target面积
    pred_widths = (pred[:, 2] - pred[:, 0] + 1.).clamp(0)
    pred_heights = (pred[:, 3] - pred[:, 1] + 1.).clamp(0)
    target_widths = (target[:, 2] - target[:, 0] + 1.).clamp(0)
    target_heights = (target[:, 3] - target[:, 1] + 1.).clamp(0)
    pred_areas = pred_widths * pred_heights
    target_areas = target_widths * target_heights

    # 求pred, target相交面积
    inter_xmins = torch.maximum(pred[:, 0], target[:, 0])
    inter_ymins = torch.maximum(pred[:, 1], target[:, 1])
    inter_xmaxs = torch.minimum(pred[:, 2], target[:, 2])
    inter_ymaxs = torch.minimum(pred[:, 3], target[:, 3])
    inter_widths = torch.clamp(inter_xmaxs - inter_xmins + 1.0, min=0.)
    inter_heights = torch.clamp(inter_ymaxs - inter_ymins + 1.0, min=0.)
    inter_areas = inter_widths * inter_heights

    # 求iou
    unions = pred_areas + target_areas - inter_areas + eps
    ious = torch.clamp(inter_areas / unions, min=eps)

    # 求最小外接矩形对角线距离
    outer_xmins = torch.minimum(pred[:, 0], target[:, 0])
    outer_ymins = torch.minimum(pred[:, 1], target[:, 1])
    outer_xmaxs = torch.maximum(pred[:, 2], target[:, 2])
    outer_ymaxs = torch.maximum(pred[:, 3], target[:, 3])
    outer_diag = torch.clamp((outer_xmaxs - outer_xmins + 1.), min=0.) ** 2 + \
        torch.clamp((outer_ymaxs - outer_ymins + 1.), min=0.) ** 2 + eps

    # 求pred与target框的中心距离
    c_pred = ((pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2)
    c_target = ((target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2)
    distance = (c_pred[0] - c_target[0] + 1.) ** 2 + (c_pred[1] - c_target[1] + 1.) ** 2

    # 求diou loss
    dious = ious - distance / outer_diag
    if reduce == 'mean':
        loss = torch.mean(1 - dious)
    elif reduce == 'sum':
        loss = torch.sum(1 - dious)
    else:
        raise NotImplementedError

    return loss


def ciou_loss(pred, target, reduce='mean', eps=1e-6):
    """
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction: "mean" or "sum"
    return: loss
    """
    # 求pred, target面积
    pred_widths = (pred[:, 2] - pred[:, 0] + 1.).clamp(0)
    pred_heights = (pred[:, 3] - pred[:, 1] + 1.).clamp(0)
    target_widths = (target[:, 2] - target[:, 0] + 1.).clamp(0)
    target_heights = (target[:, 3] - target[:, 1] + 1.).clamp(0)
    pred_areas = pred_widths * pred_heights
    target_areas = target_widths * target_heights

    # 求pred, target相交面积
    inter_xmins = torch.maximum(pred[:, 0], target[:, 0])
    inter_ymins = torch.maximum(pred[:, 1], target[:, 1])
    inter_xmaxs = torch.minimum(pred[:, 2], target[:, 2])
    inter_ymaxs = torch.minimum(pred[:, 3], target[:, 3])
    inter_widths = torch.clamp(inter_xmaxs - inter_xmins + 1.0, min=0.)
    inter_heights = torch.clamp(inter_ymaxs - inter_ymins + 1.0, min=0.)
    inter_areas = inter_widths * inter_heights

    # 求iou
    unions = pred_areas + target_areas - inter_areas + eps
    ious = torch.clamp(inter_areas / unions, min=eps)

    # 求最小外接矩形对角线距离
    outer_xmins = torch.minimum(pred[:, 0], target[:, 0])
    outer_ymins = torch.minimum(pred[:, 1], target[:, 1])
    outer_xmaxs = torch.maximum(pred[:, 2], target[:, 2])
    outer_ymaxs = torch.maximum(pred[:, 3], target[:, 3])
    outer_diag = torch.clamp((outer_xmaxs - outer_xmins + 1.), min=0.) ** 2 + \
        torch.clamp((outer_ymaxs - outer_ymins + 1.), min=0.) ** 2 + eps

    # 求pred与target框的中心距离
    c_pred = ((pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2)
    c_target = ((target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2)
    distance = (c_pred[0] - c_target[0] + 1.) ** 2 + (c_pred[1] - c_target[1] + 1.) ** 2

    # 求预测框形状上的损失
    w_pred, h_pred = pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1] + eps
    w_target, h_target = target[:, 2] - target[:, 0], target[:, 3] - target[:, 1] + eps
    factor = 4 / (math.pi ** 2)
    v = factor * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_target / h_target), 2)
    alpha = v / (1 - ious + v)

    # 求ciou loss
    cious = ious - distance / outer_diag - alpha * v
    if reduce == 'mean':
        loss = torch.mean(1 - cious)
    elif reduce == 'sum':
        loss = torch.sum(1 - cious)
    else:
        raise NotImplementedError

    return loss

def focal_iou_loss(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False,  EIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=3, eps=1e-7):
    # Returns tsqrt_he IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter/union + eps, alpha)
    # beta = 2 * alpha
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha  # center distance
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                # return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # c_area = cw * ch + eps  # convex area
            # return iou - (c_area - union) / c_area  # GIoU
            c_area = torch.max(cw * ch + eps, union) # convex area
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU
    else:
        return iou # torch.log(iou+eps) or iou