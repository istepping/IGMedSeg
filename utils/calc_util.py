import numpy as np
import utils.util as util
import time


def calc_drag_point(control_pos, gt):
    """
    模拟拖拽点: 计算control_pos到gt的距离找到最远距离点P1-> 计算gt上与P1最近的距离点P2-> P1到P2即为模拟的拖拽
    """
    dist = []
    for pos in control_pos:
        dist.append(util.calc_contour_distance(gt, pos))
    indexes = sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)  # 从小到大排序存储对应索引
    start_point = control_pos[indexes[-1]]
    dist = []
    for pos in gt:
        dist.append(util.calc_distance(start_point, pos))
    indexes = sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)  # 从小到大排序存储对应索引
    end_point = gt[indexes[0]]
    return start_point, end_point


def calc_interactive_point(contour, gt):
    """
    模拟交互点: 这里的gt是按照simple提取出来的,gt上的点不规则,需要重新拟合一下然后提取
    """
    # gt_contour = util.fit_b_spline_with_geomdl(gt.copy(), interpolate=True, degree=1)
    dist = []
    for pos in gt:
        dist.append(util.calc_contour_distance(contour, pos))
    indexes = sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)  # 从小到大排序存储对应索引
    return gt[indexes[-1]]


def calc_extreme_points(contour):
    """
    计算轮廓四个极值点
    """
    contour = np.array(contour)
    top_most = list(contour[contour[:, 1].argmin()])
    left_most = list(contour[contour[:, 0].argmin()])
    bottom_most = list(contour[contour[:, 1].argmax()])
    right_most = list(contour[contour[:, 0].argmax()])
    return [top_most, left_most, bottom_most, right_most]
