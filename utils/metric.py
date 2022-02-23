import numpy as np
from skimage.draw import polygon
import torch
import cv2

"""
1. [IoU相关计算](https://blog.csdn.net/u010598525/article/details/108667524)

"""


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


"""
公式+代码
"""


def calc_iou_with_polygon(polygon_1, polygon_2):
    """
    计算两个多边形的IoU,polygon=np.array([[x,y],[x,y],....])
    [参考](https://blog.csdn.net/qq_40081208/article/details/105309319)
    """
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


"""
计算mask的iou
"""


def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1 + mask2) == 2).sum()
    iou = inter / (area1 + area2 - inter)
    return iou


# batch_size
def mask_iou_pytorch(mask1, mask2):
    """
    mask1: [m1,n] m1 means number of predicted objects
    mask2: [m2,n] m2 means number of gt objects
    Note: n means image_w x image_h
    """
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection
    iou = intersection / union
    return iou


# 单目标mask_iou计算速度太慢-> 请使用mask_iou
def single_mask_iou_pytorch(mask1, mask2):
    mask1.gt_(0)
    mask2.gt_(0)
    mask1 = mask1.view(1, -1).squeeze()
    mask2 = mask2.view(1, -1).squeeze()
    intersection = 0
    for i in range(mask1.shape[0]):
        if mask1[i] == 1 and mask2[i] == 1:
            intersection += 1
    area1 = torch.sum(mask1).item()
    area2 = torch.sum(mask2).item()
    union = (area1 + area2) - intersection
    iou = round(intersection / union, 3)
    return iou


"""
Dice = 2 * (A∩B) /(|A| + |B|)
"""


def dice(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum()
    if union != 0:
        dices = float((2 * intersection) / union)
    else:
        dices = 0
    return dices


"""
HD: 默认使用欧氏距离, 计算两个mask的HD
Hausdorff Distance: HD, 豪斯多夫距离, 度量空间中任意两个集合之间定义的一种距离
[HD距离细节](https://blog.csdn.net/qq_39575835/article/details/96460935)
"""






"""
RVD= (A-B)/B
A: pre
B: Gt
"""


def rvd(mask1, mask2):
    A = (mask1 == 1).sum()
    B = (mask2 == 1).sum()
    return (A - B) / B


"""
ASSD
"""


if __name__ == "__main__":
    # mask:保存0,1元素
    m1 = np.array([1, 1, 1, 1])
    m2 = np.array([[0, 1], [1, 0], [0, 0], [0, 0]])
    # print(dice(m1, m2))
    print(np.array(np.where(m2 == 1)))
