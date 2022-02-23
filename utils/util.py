"""
@Time 2020/3/13
@Author Rocky
@Note 工具包
"""

import numpy as np
import pygame
import torch
from PIL import Image
from geomdl import fitting
import os
import utils.metric as metric
import json
import utils.screen_draw as screen_draw
from common.common import *
from model.snake import Snake
import math


# 初始化新的分割任务
def init_seg():
    # MedSeg
    CONTROL_POS.clear()
    INIT_CONTROL_POS.clear()
    ORIGIN_IMAGE_SIZE.clear()
    IMAGE_NAME.clear()
    INITIAL_POINTS.clear()
    INTERACTIVE_POINT.clear()
    TRUE_CONTROL_INDEX.clear()


def get_learning_labels_and_idx(initial_control_pos, control_pos):
    # 交互点与初始点:
    train_idx = []
    labels = np.zeros((len(initial_control_pos), 2))
    for i, point in enumerate(control_pos):
        train_idx.append(i)
        shift_x = point[0] - initial_control_pos[i][0]
        shift_y = point[1] - initial_control_pos[i][1]
        labels[i][0] = shift_x
        labels[i][1] = shift_y
    return torch.tensor(labels, dtype=torch.float), torch.tensor(train_idx, dtype=torch.long)


def get_range_k(N, inum):
    # N // 4 - inum * 0.35
    return max(0, N // 4 - math.floor(inum * 0.35))


def exclude_re_points(control_pos):
    points = []
    for p in control_pos:
        if p not in points:
            points.append(p)
    control_pos.clear()
    control_pos.extend(points)


def exclude_points(control_pos):
    """
    排除异常点, 修改CONTROL_POS值
    """
    for i, point in enumerate(control_pos):
        dist1 = calc_distance(point, control_pos[(i - 1) % len(control_pos)])
        dist2 = calc_distance(point, control_pos[(i + 1) % len(control_pos)])
        dist3 = 2 * calc_distance(control_pos[(i - 1) % len(control_pos)], control_pos[(i + 1) % len(control_pos)])
        if dist1 >= dist3 or dist2 >= dist3:
            control_pos.pop(i)
            print("=>exclude point=", point)


def snake(current_img, screen, control_pos, iter_num=20, closed=True, show=True, search_kernel_size=7):
    img = pygame_to_cv(current_img)
    sk = Snake(img, closed=closed, points=control_pos, search_kernel_size=search_kernel_size)
    for i in range(iter_num):
        sk.step()
        # np.array(sk.points).tolist()
        if show:
            contour = fit_b_spline_with_geomdl(np.array(sk.points).tolist().copy(), interpolate=INTERPOLATE)
            screen_draw.show_and_cal(screen, current_img, contour, sk.points.copy())

    return np.array(sk.points)


def auto_generate_control_pos(contour, control_pos, point, s=10, k=2):
    index = get_near_control_pos(contour, point)
    # 以contour[index]为中心, 以s为步长周围采样k个拓展点, 并且存储到control_pos中: 使用删除的思路删除步长内部的控制点
    indexes = []
    # 以删除的思路, 从idx左侧到idx范围中间删除,右侧遍历到 K+1
    # 多索引删除操作,注意list大小变化: 方法: 记录删除索引, 按照索引生成新的list(不含有旧的索引),
    for i in range(-k, k + 2):
        idx = (index + i * s) % len(contour)
        left = (index + (i - 1) * s) % len(contour)
        # 删除操作
        if left < idx:
            for ix in range(left + 1, idx):
                indexes.append(ix)
        else:
            for ix in range(0, idx):
                indexes.append(ix)
            for ix in range(left + 1, len(contour)):
                indexes.append(ix)
    control_pos.clear()
    control_pos.extend([list(contour[i]) for i in range(len(contour)) if i not in indexes])


def auto_generate_graph(contour, control_pos, point):
    """
    生成两个控制点
    """
    index = get_near_control_pos(contour, point)
    all_index = []
    for p in control_pos:
        all_index.append(get_near_control_pos(contour, p))
    all_index.append(index)
    all_index.sort()

    # 插入数据,all_index存储control_pos对应点在contour上的位置
    idx = all_index.index(index)

    # 记录交互前精度
    # control_pos.insert(idx, contour[index])
    # contour = fit_b_spline_with_geomdl(INIT_CONTROL_POS.copy(), interpolate=INTERPOLATE)
    # if os.path.exists(GT_JSON_PATH[-1]):
    #     poly = json.load(open(f"{GT_JSON_PATH[-1]}"))["polys"][0]["poly"]  # GT
    #     print("Spline-多边形精度(IoU)=", metric.calc_iou_with_polygon(np.array(poly), np.array(control_pos)))
    #     print("Spline-样条曲线精度(IoU)=", metric.calc_iou_with_polygon(np.array(poly), np.array(contour)))
    # control_pos.pop(idx)

    if idx + 1 >= len(all_index):
        control_pos.insert(idx, contour[(all_index[idx] + len(contour - 1)) // 2])
    else:
        control_pos.insert(idx, contour[(all_index[idx] + all_index[idx + 1]) // 2])
    control_pos.insert(idx, contour[index])
    control_pos.insert(idx, contour[(all_index[idx] + all_index[idx - 1]) // 2])

    # 准备模型输入
    index = get_near_control_pos(control_pos, point)

    input_points = []
    inter_index = []
    link_range = 2
    for i in range(-link_range, link_range + 1):
        idx = (index + i) % len(control_pos)
        inter_index.append(idx)
        input_points.append(control_pos[idx])

    num_points = link_range * 2 + 1
    train_idx = [0, 2, 4]
    labels = np.zeros((num_points, 2))
    shift_x, shift_y = point[0] - control_pos[index][0], point[1] - control_pos[index][1]
    labels[2][0], labels[2][1] = shift_x, shift_y
    labels[0][0], labels[0][1] = 0, 0
    labels[4][0], labels[4][1] = 0, 0

    return inter_index, torch.tensor(input_points, dtype=torch.long), torch.tensor(labels,
                                                                                   dtype=torch.float), torch.tensor(
        train_idx, dtype=torch.long)


def fit_b_spline_with_geomdl(points, interpolate=True, degree=2, closed=True):
    # The NURBS Book Ex9.1
    ptx = points.copy()
    points = []
    for p in [list(i) for i in ptx]:
        if p not in points:
            points.append(p)
    if closed:
        points.append(points[0])  # 曲线闭合
    degree = degree  # cubic curve

    # Do global curve approximation
    if interpolate:
        try:
            curve = fitting.interpolate_curve(points, degree)
        except:
            print(points)
    else:
        curve = fitting.approximate_curve(points, degree)
    # Plot the interpolated curve
    curve.delta = 0.01  # 0.005
    # curve.vis = vis.VisCurve2D()
    # curve.render()

    # Prepare points
    evalpts = np.array(curve.evalpts)
    # pts = np.array(points)
    # # Plot points together on the same graph
    # fig = plt.figure(figsize=(10, 8), dpi=96)
    # plt.plot(evalpts[:, 0], evalpts[:, 1])
    # plt.scatter(pts[:, 0], pts[:, 1], color="red")
    # plt.show()
    evalpts = np.asarray(evalpts, dtype=np.int32)
    points = []
    for p in [list(i) for i in evalpts]:
        if p not in points:
            points.append(p)
    return np.asarray(points, dtype=np.int32)


def get_insert_index(contour, points, point):
    index = get_near_control_pos(contour, point)
    all_index = []
    for p in points:
        all_index.append(get_near_control_pos(contour, p))
    all_index.append(index)
    all_index.sort()
    idx = all_index.index(index)
    return idx


# 获取最近的点,但是不包括已经交互点
def get_near_control_pos(control_pos, point):
    dist = []
    for pos in control_pos:
        dist.append(calc_distance(point, pos))
    indexes = sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)  # 从小到大排序
    index = indexes[0]
    # 返回最小的对应的索引,但是没有被用户确定的点
    # for i in indexes:
    #     if i not in TRUE_CONTROL_INDEX:
    #         index = i
    #         break
    return index


def get_label(poly, control_pos):
    num_points = len(control_pos)
    label = np.zeros((num_points, 2))
    for i, point in enumerate(control_pos):
        dist = []
        for pos in poly:
            dist.append(calc_distance(pos, point))
        indexes = sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)  # 从小到大排序存储对应索引
        end_point = poly[indexes[0]]
        label[i][0], label[i][1] = end_point[0] - point[0], end_point[1] - point[1]
    return torch.tensor(label, dtype=torch.float)


# 交互式输入
def get_drag_interactive_input(control_pos, start_point, end_point, link_range=1):
    control_pos = [list(i) for i in control_pos]
    index = control_pos.index([i for i in start_point])

    input_points = []
    inter_index = []
    for i in range(-link_range, link_range + 1):
        idx = (index + i) % len(control_pos)
        inter_index.append(idx)
        input_points.append(control_pos[idx])

    num_points = link_range * 2 + 1
    train_idx = [int(num_points // 2)]
    labels = np.zeros((num_points, 2))
    shift_x, shift_y = end_point[0] - control_pos[index][0], end_point[1] - control_pos[index][1]
    labels[int(num_points // 2)][0], labels[int(num_points // 2)][1] = shift_x, shift_y

    return inter_index, torch.tensor(input_points, dtype=torch.long), torch.tensor(labels,
                                                                                   dtype=torch.float), torch.tensor(
        train_idx, dtype=torch.long)


def get_interactive_seg_input(control_pos, point, link_range=1):
    index = get_near_control_pos(control_pos, point)

    input_points = []
    inter_index = []
    for i in range(-link_range, link_range + 1):
        idx = (index + i) % len(control_pos)
        inter_index.append(idx)
        input_points.append(control_pos[idx])

    num_points = link_range * 2 + 1
    train_idx = [int(num_points // 2)]
    labels = np.zeros((num_points, 2))
    shift_x, shift_y = point[0] - control_pos[index][0], point[1] - control_pos[index][1]
    labels[int(num_points // 2)][0], labels[int(num_points // 2)][1] = shift_x, shift_y

    return inter_index, torch.tensor(input_points, dtype=torch.long), torch.tensor(labels,
                                                                                   dtype=torch.float), torch.tensor(
        train_idx, dtype=torch.long)


# 准备模型输入信息
def get_input_img(img):
    if isinstance(img, np.ndarray):
        """cv2"""
        img = cv_to_pil(img)
    else:
        img = pygame_to_pil(img)

    return img


def get_input_points(points):
    return torch.tensor(points, dtype=torch.long)


def get_input_labels_and_train_idx(initial_points, init_control_pos):
    # 交互点与初始点:
    train_idx = []
    labels = np.zeros((len(init_control_pos), 2))
    for point in initial_points:
        dist = []
        for pos in init_control_pos:
            dist.append(calc_distance(point, pos))
        idx = sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)  # 返回排序索引,从小到大,idx[0]就是最近距离点对应的索引
        train_idx.append(idx[0])
        shift_x = point[0] - init_control_pos[idx[0]][0]
        shift_y = point[1] - init_control_pos[idx[0]][1]
        labels[idx[0]][0] = shift_x
        labels[idx[0]][1] = shift_y
    return torch.tensor(labels, dtype=torch.float), torch.tensor(train_idx, dtype=torch.long)


# 计算两个坐标点的距离
def calc_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calc_contour_distance(contour, point):
    """计算点到多边形轮廓距离,返回绝对值"""
    contour = np.array(contour)
    dist = cv2.pointPolygonTest(contour, tuple(point), measureDist=True)

    return abs(round(dist, 2))


# 初始轮廓拟合
def get_convexHull_from_initial_points(points, image_size):
    contour = np.array(points)
    hull = cv2.convexHull(contour)
    hull = shrink_polygon(hull.squeeze(1), -0.6)
    hull = hull[:, np.newaxis, :]
    img = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    cv2.polylines(img, [hull], color=(255, 255, 255), isClosed=True, thickness=1)  # cnt外加上[]才能形成多边形
    # cv_img_show(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, APPROX)
    contour = contours[0].squeeze()
    return contour


def get_ellipse_from_initial_points(points, image_size):
    contour = np.array(points)
    img = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    box = cv2.boundingRect(contour)
    ellipse = cv2.fitEllipse(box)
    # cv2.minAreaRect(contour) 最小矩形,cv2.boundingRect(contour) 正矩形
    cv2.ellipse(img, ellipse, color=(255, 255, 255), thickness=1)
    # cv2.polylines(img, ellipse, color=(255, 255, 255), isClosed=True,lineType=1)
    # [参数设置](https://blog.csdn.net/hjxu2016/article/details/77833336/) cv2.CHAIN_APPROX_NONE,cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, APPROX)
    contour = contours[0].squeeze()
    # print(contour)
    return contour


# 面积
def Area(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    area = 0.
    vector_1 = polygon[1] - polygon[0]
    for i in range(2, N):
        vector_2 = polygon[i] - polygon[0]
        area += np.abs(np.cross(vector_1, vector_2))
        vector_1 = vector_2
    return area / 2


# 多边形周长
# shape of polygon: [N, 2]
def Perimeter(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    permeter = 0.
    for i in range(N):
        permeter += np.linalg.norm(polygon[i - 1] - polygon[i])
    return permeter


# |r| < 1
# r > 0, 内缩
# r < 0, 外扩
def calc_shrink_width(polygon: np.array, r):
    area = Area(polygon)
    perimeter = Perimeter(polygon)
    L = area * (1 - r ** 2) / perimeter
    return L if r > 0 else -L


def shrink_polygon(polygon: np.array, r):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    shrinked_polygon = []
    L = calc_shrink_width(polygon, r)
    for i in range(N):
        Pi = polygon[i]
        v1 = polygon[i - 1] - Pi
        v2 = polygon[(i + 1) % N] - Pi

        normalize_v1 = v1 / np.linalg.norm(v1)
        normalize_v2 = v2 / np.linalg.norm(v2)

        sin_theta = np.abs(np.cross(normalize_v1, normalize_v2))

        Qi = Pi + L / sin_theta * (normalize_v1 + normalize_v2)
        shrinked_polygon.append(Qi)
    return np.asarray(shrinked_polygon, dtype=np.int32)


# box->　生成拟合椭圆
def get_ellipse_from_box(box, image_size):
    # cnt = np.array(box)
    if len(INITIAL_POINTS) > 4:
        # 使用点集合拟合
        cnt = np.array(INITIAL_POINTS)
    else:
        points = []
        # 矩形框采样形成轮廓->
        for x in range(box[0][0], box[1][0], 2):
            points.append([x, box[0][1]])
        for y in range(box[0][1], box[1][1], 2):
            points.append([box[1][0], y])
        for x in reversed(range(box[0][0], box[1][0], 2)):
            points.append([x, box[1][1]])
        for y in reversed(range(box[0][1], box[1][1], 2)):
            points.append([box[0][0], y])
        cnt = np.array(points)
    # ellipse = cv2.approxPolyDP(cnt, epsilon=15, closed=True)
    # ellipse = cv2.fitEllipse(cnt)
    ellipse = cv2.fitEllipse(cnt)
    # 获取椭圆坐标
    img = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    cv2.ellipse(img, ellipse, (255, 255, 255), 1)
    # cv_img_show(img)
    # cv2.polylines(img, ellipse, color=(255, 255, 255), isClosed=True,lineType=1)
    # [参数设置](https://blog.csdn.net/hjxu2016/article/details/77833336/) cv2.CHAIN_APPROX_NONE,cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, APPROX)
    contour = contours[0].squeeze()
    # print(contour)
    return contour


# initial points-> 生成bounding box
def get_box_from_initial_points(points, expand=5, image_size=(320, 320)):
    contour = np.array(points)
    lef_top = [int(contour.min(axis=0)[0]), int(contour.min(axis=0)[1])]
    right_bottom = [int(contour.max(axis=0)[0]), int(contour.max(axis=0)[1])]
    # 进行范围拓展
    lef_top, right_bottom = [max(lef_top[0] - expand, 0), max(0, lef_top[1] - expand)], [
        min(right_bottom[0] + expand, image_size[0] - 1), min(right_bottom[1] + expand, image_size[1] - 1)]
    return [lef_top, right_bottom]


def get_mini_rect_from_initial_points(points):
    contour = np.array(points)
    center, size, angle = cv2.minAreaRect(contour)
    vertices = cv2.boxPoints((center, size, angle))
    print(vertices)

    return vertices


def export():
    mask = np.zeros((ORIGIN_IMAGE_SIZE[0][1], ORIGIN_IMAGE_SIZE[0][0]), np.uint8)
    contour = fit_b_spline_with_geomdl(CONTROL_POS.copy(), interpolate=True)
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    img = Image.fromarray(mask)
    img.save("D:\\datasets\\Few-Shot-MedSeg-RL-Based\\npc1\\R-visual\\" + IMAGE_NAME[-1] + "-" + str(
        len(INTERACTIVE_POINT)) + ".png", "png")


def pygame_to_array(img):
    return pygame.surfarray.array3d(img)


# pygame-> pil
def pygame_to_pil(img):
    img = pygame.surfarray.array3d(img)
    img = cv2.transpose(img)
    return Image.fromarray(img)


# pygame-> cv
def pygame_to_cv(img):
    img = pygame.surfarray.array3d(img)
    img = cv2.transpose(img)
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    return img


# cv-> pil
def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# cv->pygame
def cv_to_pygame(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = pygame.surfarray.make_surface(img)  # 转为pygame中surface
    img = pygame.transform.rotate(img, -90)  # 逆旋转90°
    img = pygame.transform.flip(img, True, False)  # x轴反转,y轴不变
    return img


# cv2显示图像
def cv_img_show(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 增加box_padding
def add_box_padding(size, box, padding):
    left, top = max(box[0][0] - padding, 0), max(box[0][1] - padding, 0)
    right, bottom = min(box[1][0] + padding, size[1] - 1), min(box[1][1] + padding, size[0] - 1)
    return [[left, top], [right, bottom]]


def get_shift_point(x, y):
    assert len(CONTROL_POS) > 0, "error SEG_POINTS"
    points = CONTROL_POS
    index = -1
    for i, point in enumerate(points):
        if point[0] - 1 <= x <= point[0] + 1 and point[1] - 1 <= y <= point[1] + 1:
            index = i
            return index
    return index


def get_interactive_seg_input2(control_pos, start_index, point, link_range=1):
    index = start_index
    input_points = []
    inter_index = []
    for i in range(-link_range, link_range + 1):
        idx = (index + i) % len(control_pos)
        inter_index.append(idx)
        input_points.append(control_pos[idx])

    num_points = link_range * 2 + 1
    train_idx = [int(num_points // 2)]
    labels = np.zeros((num_points, 2))
    shift_x, shift_y = point[0] - control_pos[index][0], point[1] - control_pos[index][1]
    labels[int(num_points // 2)][0], labels[int(num_points // 2)][1] = shift_x, shift_y

    return inter_index, torch.tensor(input_points, dtype=torch.long), torch.tensor(labels,
                                                                                   dtype=torch.float), torch.tensor(
        train_idx, dtype=torch.long)


if __name__ == '__main__':
    # get_concexHull_from_initial_points([[0, 0], [10, 10], [10, 11]])
    print("")
