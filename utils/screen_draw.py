import pygame
import cv2
from common.common import *
import numpy as np
import json
import os
import utils.metric as metric


def show_and_cal(screen, current_img, contour, control_pos):
    control_pos = control_pos.copy()
    if VISION_WAY == VISION_POINT:
        draw_multi_points(screen, current_img, control_pos, color=(255, 0, 0), clear=True)
    elif VISION_WAY == VISION_LINE:
        draw_pred_result(screen, current_img, contour, color=(255, 0, 0), clear=True)
    else:
        draw_multi_points(screen, current_img, control_pos, color=(255, 0, 0), clear=True)
        draw_pred_result(screen, current_img, contour, color=(255, 0, 0), clear=False)
    # if os.path.exists(GT_JSON_PATH[-1]):
    #     poly = json.load(open(f"{GT_JSON_PATH[-1]}"))["polys"][0]["poly"]  # GT
        # print("多边形精度(IoU)=", metric.calc_iou_with_polygon(np.array(poly), np.array(control_pos)))
        # print("样条曲线精度(IoU)=", metric.calc_iou_with_polygon(np.array(poly), np.array(contour)))
    pygame.display.update()


# 绘制多边形结果
def draw_pred_result(screen, current_img, points, color=(255, 0, 0), clear=True):
    if clear:
        screen.blit(current_img, IMAGE_POS)
        draw_gt(screen)
    draw_polygon(screen, points, color=color)


# 绘制多个点
def draw_multi_points(screen, current_img, points, color=(255, 0, 0), clear=True, r=2):
    if clear:
        screen.blit(current_img, IMAGE_POS)
        draw_gt(screen)
    for point in points:
        pos = [point[0] + IMAGE_POS[0], point[1] + IMAGE_POS[1]]
        pygame.draw.circle(screen, color, pos, r)


# 绘制GT
def draw_gt(screen):
    if os.path.exists(GT_JSON_PATH[-1]):
        polys = json.load(open(f"{GT_JSON_PATH[-1]}"))["polys"]  # GT
        for poly in polys:
            draw_polygon(screen, poly["poly"], color=(0, 255, 0))


# 绘制边界
def draw_polygon(screen, contour, color=(255, 0, 0)):
    cnt = []
    for point in contour:
        cnt.append([point[0] + IMAGE_POS[0], point[1] + IMAGE_POS[1]])
    pygame.draw.polygon(screen, color, np.array(cnt), 2)


# 界面重置加载部分
def init_screen(screen):
    # 加载图片资源
    background_image = pygame.image.load(background_image_filename)
    button_image = pygame.image.load(button)
    # 渲染静态界面
    # 显示图片作为背景,显示图片和坐标,划线设计布局
    screen.blit(background_image, (0, 0))
    # 标题
    text(screen, (400, 20), "An Efficient Interactive Segmentation Framework for Medical Images Without Pre-Training", 18, (0, 0, 255))
    # 设置控件
    pygame.draw.line(screen, (255, 0, 0), (0, 50), (800, 50), 1)
    pygame.draw.line(screen, (255, 0, 0), (600, 50), (600, 550), 1)
    pygame.draw.line(screen, (255, 0, 0), (0, 550), (800, 550), 1)
    screen.blit(button_image, (BUTTON_X, BUTTON_Y))  # 650,110
    text(screen, (BUTTON_X + 50, BUTTON_Y + 40), "Example", 14, (139, 136, 120))  # 700,150
    screen.blit(button_image, (BUTTON_X, BUTTON_Y + 50))
    text(screen, (BUTTON_X + 50, BUTTON_Y + 90), "Open", 14, (139, 136, 120))
    screen.blit(button_image, (BUTTON_X, BUTTON_Y + 100))
    text(screen, (BUTTON_X + 50, BUTTON_Y + 140), "Withdraw", 14, (139, 136, 120))
    screen.blit(button_image, (BUTTON_X, BUTTON_Y + 150))
    text(screen, (BUTTON_X + 50, BUTTON_Y + 190), "Done", 14, (139, 136, 120))
    screen.blit(button_image, (BUTTON_X, BUTTON_Y + 200))
    text(screen, (BUTTON_X + 50, BUTTON_Y + 240), "Finish", 14, (0, 255, 0))
    # 设置图片区域
    text(screen, (300, 300), "Image", 40, (139, 136, 120))


def show_seg(img, points):
    for i in range(len(points)):
        if i == 0:
            cv2.circle(img, (points[i][0], points[i][1]), 3, (0, 0, 255), -1)
        else:
            cv2.circle(img, (points[i][0], points[i][1]), 3, (0, 255, 0), -1)
        if i < len(points) - 1:
            cv2.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), (255, 0, 0), 1)
        else:
            cv2.line(img, (points[i][0], points[i][1]), (points[0][0], points[0][1]), (255, 0, 0), 1)
    return img


# 将控制点连成线img:array3d图像points:控制点集合
def draw_line(img, points):
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), (255, 0, 0), 1)
        else:
            cv2.line(img, (points[i][0], points[i][1]), (points[0][0], points[0][1]), (255, 0, 0), 1)
    return img


# 界面，内容，字体大小和颜色,位置
def text(screen, pos, message, size, color):
    font = pygame.font.SysFont("SimHei", size)  # 字体和大小
    new_text = font.render(message, True, color)
    text_rect = new_text.get_rect()
    text_rect.center = pos
    screen.blit(new_text, text_rect)


# 底部状态栏
def note(screen, message):
    new_message = message
    old_message = "                                              "  # 用于覆盖原位置
    font = pygame.font.SysFont("SimHei", 16)
    new_text = font.render(new_message, True, (139, 136, 120))
    old_text = font.render(old_message, True, (139, 136, 120))
    text_rect = new_text.get_rect()
    text_rect.center = (400, 550)
    old_rect = old_text.get_rect()
    old_rect.center = (400, 550)
    screen.blit(old_text, old_rect)
    screen.blit(new_text, text_rect)
