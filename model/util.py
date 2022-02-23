import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2


def get_feature_and_adj(control_pos, img_pil, points, state_size=(32, 32), link_range=3, in_channels=4):
    # 节点特征:点坐标周围区域32*32截取,Adj四路相连图结构
    feature = []
    new_image = get_pil_polygon_with_pil(img_pil.size, control_pos)
    for point in points:
        img_crop = img_pil.crop((point[0].item() - state_size[0] // 2, point[1].item() - state_size[1] // 2, point[0].item() + state_size[0] // 2, point[1].item() + state_size[1] // 2))
        img_state = transforms.ToTensor()(img_crop)
        if in_channels == 4:
            polygon_crop = new_image.crop((point[0].item() - state_size[0] // 2, point[1].item() - state_size[1] // 2, point[0].item() + state_size[0] // 2, point[1].item() + state_size[1] // 2))
            # polygon_crop.show()
            polygon_state = transforms.ToTensor()(polygon_crop)
            state = torch.cat((img_state, polygon_state), 0)
            feature.append(state.unsqueeze(0))
        else:
            feature.append(img_state.unsqueeze(0))
    feature = torch.cat(tuple(feature), 0).float()

    # 构建邻接矩阵:自环连接,周围link_range范围连接
    N = len(points)
    adj = np.zeros([N, N])
    for i in range(N):
        for j in range(-link_range, link_range + 1):
            adj[i][(i + j) % N] = 1
            adj[(i + j) % N][i] = 1
    # 使用度矩阵进行归一化
    d = np.sum(adj, axis=0)
    d = torch.tensor(np.diag(d), dtype=torch.float)
    d = torch.inverse(d)
    adj = d.mm(torch.tensor(adj, dtype=torch.float))
    return feature, adj


def get_pil_polygon_with_pil(size, control_pos):
    new_image = Image.new("1", size, "#000000")
    drawObject = ImageDraw.Draw(new_image)
    drawObject.polygon([tuple(i) for i in control_pos], outline="white")
    return new_image


def get_pil_polygon_with_cv(size, control_pos, with_line=True):
    img = np.zeros(size, dtype=np.uint8)  # 二值图像
    if with_line:
        cv2.polylines(img, [np.array([list(i) for i in control_pos])], isClosed=True, color=(255, 255, 255), thickness=3)
    else:
        for point in control_pos:
            cv2.circle(img, tuple(point), radius=2, color=(255, 255, 255), thickness=-1)
    # cv_img_show(img)
    return cv_to_pil(img)


def get_part_feature_and_adj(control_pos, img_pil, points, state_size=(32, 32), in_channels=4):
    feature = []
    # 新建pil二值化图片-> control_pos连接可视化-> crop-> concatenate
    new_image = get_pil_polygon_with_cv(img_pil.size, control_pos, with_line=False)
    # new_image = get_pil_polygon_with_pil(img_pil.size, control_pos)
    # index = 0
    for point in points:
        img_crop = img_pil.crop((point[0].item() - state_size[0] // 2, point[1].item() - state_size[1] // 2, point[0].item() + state_size[0] // 2, point[1].item() + state_size[1] // 2))
        img_state = transforms.ToTensor()(img_crop)
        # img_crop.save(f"img{index}.png")
        if in_channels == 4 or in_channels == 2:
            polygon_crop = new_image.crop((point[0].item() - state_size[0] // 2, point[1].item() - state_size[1] // 2, point[0].item() + state_size[0] // 2, point[1].item() + state_size[1] // 2))
            # polygon_crop.save(f"polygon{index}.png")
            # polygon_crop.show()
            polygon_state = transforms.ToTensor()(polygon_crop)
            state = torch.cat((img_state, polygon_state), 0)
            feature.append(state.unsqueeze(0))
        else:
            feature.append(img_state.unsqueeze(0))
        # index += 1
    feature = torch.cat(tuple(feature), 0).float()

    # 构建邻接矩阵:周围点相连+中间点连接周围所有点
    N = len(points)
    adj = np.zeros([N, N])
    for i in range(N):
        adj[i][i] = 1  # 自环
        adj[N // 2][i] = 1  # 单方向连接效果非常好
        if i + 1 < N:
            adj[i][i + 1] = 1  # 相邻连接
            adj[i + 1][i] = 1

    # 使用度矩阵进行归一化
    d = np.sum(adj, axis=0)
    d = torch.tensor(np.diag(d), dtype=torch.float)
    d = torch.inverse(d)
    adj = d.mm(torch.tensor(adj, dtype=torch.float))
    return feature, adj


def cv_to_pil(img):
    if len(img.shape) == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(img)


# cv2显示图像
def cv_img_show(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
