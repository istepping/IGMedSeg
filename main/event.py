import json
import os
import tkinter.filedialog

import pygame

import utils.calc_util as calc_util
import utils.screen_draw as screen_draw
import utils.util as util
from common.common import *


def button1(screen):
    print("1选择示例")
    util.init_seg()
    image = pygame.image.load(ex_img)
    ORIGIN_IMAGE_SIZE.append([image.get_rect()[-2], image.get_rect()[-1]])
    screen.blit(image, IMAGE_POS)
    IMAGE_NAME.append("test")

    IMAGE_PATH.append(ex_img)
    GT_JSON_PATH.append(ex_img_gt)
    screen_draw.draw_gt(screen)
    if os.path.exists(GT_JSON_PATH[-1]):
        poly = json.load(open(f"{GT_JSON_PATH[-1]}"))["polys"][0]["poly"]
        INITIAL_POINTS.extend(calc_util.calc_extreme_points(poly))
        button4(screen, image)
    return image


def button2(screen, current_img):
    print("2打开图片")
    root = tkinter.Tk()
    root.withdraw()
    file_path = tkinter.filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(DEFAULT_DIR[-1])))
    DEFAULT_DIR.append(os.path.dirname(file_path))
    if file_path != "":
        image = pygame.image.load(file_path)
        screen_draw.init_screen(screen)
        screen.blit(image, IMAGE_POS)
        util.init_seg()
        ORIGIN_IMAGE_SIZE.append([image.get_rect()[-2], image.get_rect()[-1]])
        IMAGE_NAME.append(file_path.split("/")[-1][:-4])
        IMAGE_PATH.append(file_path)
        GT_JSON_PATH.append(os.path.dirname(file_path)[
                            0:-5] + f"Label_Json/ground_truth{os.path.basename(file_path).split('.')[0][5:]}.json")
        if os.path.exists(GT_JSON_PATH[-1]):
            screen_draw.draw_gt(screen)
            poly = json.load(open(f"{GT_JSON_PATH[-1]}"))["polys"][0]["poly"]
            INITIAL_POINTS.extend(calc_util.calc_extreme_points(poly))
            button4(screen, image)
            return image, OPERATION_MODIFY
        return image, OPERATION_PRE
    return current_img, OPERATION_PRE


def button3(screen, current_img):
    print("3撤销点击")
    if len(INITIAL_POINTS) > 0:
        INITIAL_POINTS.pop()
        screen.blit(current_img, IMAGE_POS)
        for pos in INITIAL_POINTS:
            pygame.draw.circle(screen, (255, 0, 0), (pos[0] + IMAGE_POS[0], pos[1] + IMAGE_POS[1]), 1)


def button4(screen, current_img):
    print("4完成点击")
    if len(INITIAL_POINTS) < 2:
        return
    contour = util.fit_b_spline_with_geomdl(INITIAL_POINTS.copy(), interpolate=True)

    for i in range(0, len(contour), SAMPLING_STEP):
        INIT_CONTROL_POS.append(contour[i])
        CONTROL_POS.append(contour[i])
    screen_draw.show_and_cal(screen, current_img, contour, CONTROL_POS.copy())
    # print("control_pos_num=", len(CONTROL_POS))


# 进行进一步交互式分割
def get_interactive_point(screen, current_img, pos):
    x, y = pos[0] - IMAGE_POS[0], pos[1] - IMAGE_POS[1]
    if 0 <= x < ORIGIN_IMAGE_SIZE[0][0] and 0 <= y < ORIGIN_IMAGE_SIZE[0][1]:
        INTERACTIVE_POINT.append([x, y])
        inter_index, input_points, input_labels, train_idx = util.get_interactive_seg_input(CONTROL_POS,
                                                                                            INTERACTIVE_POINT[
                                                                                                -1], link_range=4)
        output = MODEL.refine_seg(CONTROL_POS, util.get_input_img(current_img), input_points, input_labels,
                                  train_idx, epoch=20, part=True)
        for i, index in enumerate(inter_index):
            if i in train_idx:
                shift_x, shift_y = input_labels[i]
            else:
                shift_x, shift_y = output[i]
            CONTROL_POS[index] = [CONTROL_POS[index][0] + int(shift_x.item()),
                                  CONTROL_POS[index][1] + int(shift_y.item())]

        contour = util.fit_b_spline_with_geomdl(CONTROL_POS.copy(), interpolate=True)
        screen_draw.show_and_cal(screen, current_img, contour, CONTROL_POS.copy())


def button5(screen):
    print("5完成分割")
    note = " " + str(len(INTERACTIVE_POINT))
    screen_draw.note(screen, note)
    util.export()


# 获取初始交互点
def get_initial_interactive_point(screen, pos):
    x, y = pos[0] - IMAGE_POS[0], pos[1] - IMAGE_POS[1]
    if 0 <= x < ORIGIN_IMAGE_SIZE[0][0] and 0 <= y < ORIGIN_IMAGE_SIZE[0][1]:
        INITIAL_POINTS.append([x, y])
        pygame.draw.circle(screen, (255, 0, 0), pos, 1)


def click(pos):
    x, y = pos[0] - IMAGE_POS[0], pos[1] - IMAGE_POS[1]
    if 0 <= x < ORIGIN_IMAGE_SIZE[0][0] and 0 <= y < ORIGIN_IMAGE_SIZE[0][1]:
        index = util.get_shift_point(x, y)
        if index < 0:
            return True
        else:
            SHIFT_INDEX.append(index)
    else:
        return True


def get_end_point(screen, current_img, pos):
    if len(SHIFT_INDEX) <= 0:
        return
    x, y = pos[0] - IMAGE_POS[0], pos[1] - IMAGE_POS[1]
    start_point = SHIFT_INDEX[-1]
    SHIFT_INDEX.clear()
    inter_index, input_points, input_labels, train_idx = util.get_interactive_seg_input2(CONTROL_POS, start_point,
                                                                                         (x, y), link_range=4)
    output = MODEL.refine_seg(CONTROL_POS, util.get_input_img(current_img), input_points, input_labels,
                              train_idx, epoch=10, part=True)
    for i, index in enumerate(inter_index):
        if i in train_idx:
            shift_x, shift_y = input_labels[i]
        else:
            shift_x, shift_y = output[i]
        CONTROL_POS[index] = [CONTROL_POS[index][0] + int(shift_x.item()),
                              CONTROL_POS[index][1] + int(shift_y.item())]
    contour = util.fit_b_spline_with_geomdl(CONTROL_POS.copy(), interpolate=True)
    screen_draw.show_and_cal(screen, current_img, contour, CONTROL_POS.copy())


def modify(screen, pos, current_img):
    x, y = pos[0] - IMAGE_POS[0], pos[1] - IMAGE_POS[1]
    points = CONTROL_POS.copy()
    points.pop(SHIFT_INDEX[0])
    points.insert(SHIFT_INDEX[0], [x, y])
    contour = util.fit_b_spline_with_geomdl(points, interpolate=True)
    screen_draw.show_and_cal(screen, current_img, contour, points)


def button6(screen, current_img):
    if SHOW[0]:
        screen.blit(current_img, IMAGE_POS)
        screen_draw.draw_gt(screen)

        SHOW.clear()
        SHOW.append(False)
    else:
        screen.blit(current_img, IMAGE_POS)

        SHOW.clear()
        SHOW.append(True)


def button7(screen, current_img):
    if SHOW[0]:
        screen_draw.draw_multi_points(screen, current_img, CONTROL_POS, color=(255, 0, 0), clear=True)
        contour = util.fit_b_spline_with_geomdl(CONTROL_POS.copy(), interpolate=True)
        screen_draw.draw_pred_result(screen, current_img, contour, color=(255, 0, 0), clear=False)

        SHOW.clear()
        SHOW.append(False)
    else:
        screen_draw.draw_multi_points(screen, current_img, CONTROL_POS, color=(255, 0, 0), clear=True, show_gt=False)
        contour = util.fit_b_spline_with_geomdl(CONTROL_POS.copy(), interpolate=True)
        screen_draw.draw_pred_result(screen, current_img, contour, color=(255, 0, 0), clear=False, show_gt=False)

        SHOW.clear()
        SHOW.append(True)
