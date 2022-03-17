import numpy as np
from skimage.draw import polygon
import torch


def get_mask(img):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    img[img >= 1] = 1
    img[img < 1] = 0
    return img


def get_bool_mask(img):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    img[img >= 1] = True
    img[img < 1] = False
    return np.array(img, dtype=bool)


def calc_iou_with_polygon(polygon_1, polygon_2):
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])
    r_max = max(rr1.max(), rr2.max()) + 1
    c_max = max(cc1.max(), cc2.max()) + 1
    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    intersection = np.sum(canvas == 2)
    if union == 0:
        return 0
    return round(intersection / union, 4)


def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1 + mask2) == 2).sum()
    iou = inter / (area1 + area2 - inter)
    return iou


def mask_iou_pytorch(mask1, mask2):
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection
    iou = intersection / union
    return iou
