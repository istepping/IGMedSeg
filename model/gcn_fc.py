import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from torch.nn import init
from torch.nn.parameter import Parameter


class GCN_FC(nn.Module):
    def __init__(self, in_channels=4, out_features=2):
        print("CGCN-init")
        # 4 * 32 * 32
        super(GCN_FC, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1))
        # fc拟合数据有用
        in_features = 64 * 2 * 2  # 32 * 6 * 6, 64 * 2 * 2
        self.fc1 = nn.Linear(in_features=in_features, out_features=in_features // 4)
        self.fc2 = nn.Linear(in_features=in_features // 4, out_features=in_features // 8)
        self.fc3 = nn.Linear(in_features=in_features // 8, out_features=out_features)
        self.relu = nn.ReLU(inplace=True)
        # self.initNetParams()

    def forward(self, x, adj):
        # print("GCN-forward")
        x = self.relu(self.conv1(x))  # 32*30*30
        x = self.max_pool(x)  # 32*7*7
        x = self.relu(self.conv2(x))  # 64*13*13
        x = self.max_pool(x)  # 64*6*6
        x = self.relu(self.conv3(x))  # 64*6*6
        x = self.max_pool(x)  # 64*2*2
        # print(x.shape)
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, 1)  # 24*392
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        # print(output)
        return output

    def initNetParams(self):
        '''Init net parameters.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight, gain=1.0)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                init.constant_(m.bias, 0)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        # print("GraphConvolution-init")
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.zeros(in_features, out_features), requires_grad=True)
        self.bias = Parameter(torch.zeros(out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # print("GraphConvolution-forward")
        # 传播规则:AHW+B
        output = torch.mm(adj, x)
        output = torch.mm(output, self.weight)
        output = output + self.bias
        return output
