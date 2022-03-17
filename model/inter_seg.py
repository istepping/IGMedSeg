import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as f
from common.common import *
import json
import os
import time
import matplotlib.pyplot as plt
import model.util as util
from model.gcn import GCN
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class InterSeg:
    def __init__(self, load_model=True, verbose=False, sgd=True, show_loss=False, in_channels=4, fc=False):
        print("InterSeg-init")
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.in_channels = in_channels
        self.model = GCN(in_channels=self.in_channels)
        self.param = json.load(open(r"../model/param.json"))
        if load_model and os.path.exists(self.param["load_url"]):
            print("==>load model from {}".format(self.param["load_url"]))
            self.model.load_state_dict(torch.load(self.param["load_url"]))
        if sgd:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.param["train"]["lr"], weight_decay=self.param["train"]["weight_decay"])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param["train"]["lr"],
                                        weight_decay=self.param["train"]["weight_decay"])
        self.train_count = 0
        self.verbose = verbose
        self.show_loss = show_loss

    def predict(self, control_pos, img_pil, points):
        print("predict with control_pos")
        node_feature, adj = util.get_feature_and_adj(control_pos, img_pil, points)
        output = self.model(node_feature, adj)
        return output

    def refine_seg(self, control_pos, img_pil, points, labels, train_idx, epoch=10, part=False):
        if self.verbose:
            print("InterSeg-refine-seg with epoch=", epoch)
            print("labels=", labels)
        # img_pil是个三通道一样的数据结构,如果in_channels<=2,则转为单通道
        if self.in_channels <= 2:
            img_pil = img_pil.convert("L")
        # 一个是完整环状图,一个是非环状调整图
        if part:
            node_feature, adj = util.get_part_feature_and_adj(control_pos, img_pil, points, in_channels=self.in_channels)
        else:
            node_feature, adj = util.get_feature_and_adj(control_pos, img_pil, points, in_channels=self.in_channels)

        # 训练,和预测
        loss_list = []
        step_list = []
        output_list = []
        self.model.train()
        time1 = time.time()
        # train
        for i in range(epoch):
            # print("epoch={}".format(i + 1))
            self.optimizer.zero_grad()
            output = self.model(node_feature, adj)
            loss = f.mse_loss(output[train_idx], labels[train_idx])
            loss_list.append(loss.item())
            step_list.append(i + 1)
            output_list.append(output)
            loss.backward(retain_graph=True)
            self.optimizer.step()
        self.train_count += 1
        # print("==> train end!")
        if self.verbose:
            print("一次交互算法运行时间(S):", round(time.time() - time1, 3))
        # 绘图
        # print(step_list)
        # print(loss_list)
        if self.show_loss:
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(step_list, loss_list)
            plt.show()
        # self.save()
        out = output_list[loss_list.index(min(loss_list))]
        if self.verbose:
            print(out)
        return out

    def normalize(self, x):
        batches = torch.unbind(x, dim=0)
        output = []
        for item in batches:
            output.append(self.normalizer(item))
        return torch.stack(output, dim=0)

    def save(self):
        # print(self.train_count)
        # if self.train_count > 0:
        torch.save(self.model.state_dict(), self.param["load_url"])  # 保存参数
        print("==>model saved in {}".format(self.param["load_url"]))
