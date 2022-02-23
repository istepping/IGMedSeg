import torch.optim as optim
from model.gcn import GCN
import matplotlib.pyplot as plt
import model.util as util
import os
from PIL import Image
import cv2
import json
import utils.calc_util as calc_util
from tests.data import get_json_from_png
import utils.util as utils
import torch
from tqdm import tqdm
import random
import torch.nn.functional as f

"""
训练数据集训练一个模型并保存
train流程: batch_size,
"""


class Trainer:
    def __init__(self, debug=True, load_model=True, load_model_path=r"../ex/model/cgcn-npc1-30000.pt", image_dir=r"D:\datasets\Lib\NPC\train\images",
                 label_dir=r"D:\datasets\Lib\NPC\train\labels_1", epoch=50000, dataset="npc1"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dataset = dataset
        self.batch_size = 16
        self.epoch = epoch
        self.count = 0
        self.debug = debug
        self.model = GCN(in_channels=4)
        if load_model and os.path.exists(load_model_path):
            print("==>load model from {}".format(load_model_path))
            self.model.load_state_dict(torch.load(load_model_path))
            self.count = int(load_model_path.split("/")[-1].split(".")[0].split("-")[-1])
            print(self.count)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.8)

    def train(self):
        print()
        # 加载数据: 生成node_features和adjs
        label_files = os.listdir(self.label_dir)
        node_features, adjs, labels = [], [], []
        print("loading dataset...")
        for image_file in tqdm(label_files):
            img_cv = cv2.imread(f"{self.image_dir}\\{image_file}")
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            gt = get_json_from_png(f"{self.label_dir}\\{image_file}", 0)
            if gt["num_cont"] < 1:
                continue
            poly = gt["polys"][0]["poly"]
            extreme_points = calc_util.calc_extreme_points(poly)
            contour = utils.fit_b_spline_with_geomdl(extreme_points.copy(), interpolate=True)
            if len(contour) < 40:
                continue
            control_pos = random.sample([list(i) for i in contour], 40)
            node_feature, adj = util.get_feature_and_adj(control_pos, img_pil, control_pos)
            label = utils.get_label(poly, control_pos)
            # print(label)

            node_features.append(node_feature.unsqueeze(0))
            adjs.append(adj.unsqueeze(0))
            labels.append(label.unsqueeze(0))
            if self.debug and len(labels) > 20:
                break
        node_features = torch.cat(tuple(node_features), 0).float()
        adjs = torch.cat(tuple(adjs), 0).float()
        labels = torch.cat(tuple(labels), 0).float()
        total = adjs.shape[0]
        print("dataset loaded!")
        print(node_features.shape)
        print(adjs.shape)
        print(labels.shape)
        # 训练流程
        loss_list = []
        step_list = []
        self.model.train()
        print("train...")
        for i in tqdm(range(self.epoch)):
            # print("epoch={}".format(i + 1))
            self.optimizer.zero_grad()
            # start, end = (i * self.batch_size) % total, ((i + 1) * self.batch_size) % total
            # if end > start:
            #     index = range(start, end)
            # else:
            #     index = list(range(start, total))
            #     index.extend(list(range(0, end)))
            # index = range(0, total)
            index = random.sample(range(0, total), self.batch_size)
            loss = torch.tensor(0., dtype=torch.float)
            for id in index:
                output = self.model(node_features[id], adjs[id])
                loss += f.mse_loss(output, labels[id])
            loss.backward()  # retain_graph=True可以保存值, 再进行backward()
            self.optimizer.step()

            if self.debug and (i + 1) % 10 == 0:
                print(f"step={i + 1},loss={round(loss.item(), 2)}")
                # print(output)
            if (i + 1) % 500 == 0 and not self.debug:
                loss_list.append(loss.item())
                step_list.append(i + 1)
                print(f"step={i + 1},loss={round(loss.item(), 2)}")
            if (i + 1) % 5000 == 0 and not self.debug:
                self.save(self.count + 1)
            self.count += 1
            self.scheduler.step()
        # 结果输出
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(step_list, loss_list)
        plt.show()
        # self.save(self.count + 1)

    def save(self, epoch):
        path = f"../ex/model/cgcn-{self.dataset}-{epoch}.pt"
        torch.save(self.model.state_dict(), path)  # 保存参数
        print("==>model saved in {}".format(path))


if __name__ == "__main__":
    trainer = Trainer(debug=False, load_model=False)
    trainer.train()
