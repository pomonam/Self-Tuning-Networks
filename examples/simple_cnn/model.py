from base_model import StnModel
from layers.conv2d import *
from layers.linear import *
from utils.dropout_utils import *

import torch.nn.functional as F
import torch.nn as nn


class StnSimpleCNN(StnModel):
    def __init__(self, num_hyper, h_container, bias=True):
        super(StnSimpleCNN, self).__init__()
        self.num_hyper = num_hyper
        self.h_container = h_container
        self.bias = bias

        self.conv1 = StnConv2d(1, 16, kernel_size=5, padding=2, num_hyper=self.num_hyper, bias=self.bias)
        self.conv2 = StnConv2d(16, 32, kernel_size=5, padding=2, num_hyper=self.num_hyper, bias=self.bias)

        self.fc1 = StnLinear(7 * 7 * 32, 7 * 7 * 32, num_hyper=self.num_hyper, bias=self.bias)
        self.fc2 = StnLinear(7 * 7 * 32, 10, num_hyper=self.num_hyper, bias=self.bias)

        self.layers = nn.ModuleList(
            [self.conv1, self.conv2, self.fc1, self.fc2]
        )

    def get_layers(self):
        return self.layers

    def forward(self, x, h_net, h_param):
        if "dropout0" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout0"), self.training)

        x = self.conv1(x, h_net)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout1" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout1"), self.training)

        x = self.conv2(x, h_net)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout2" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout2"), self.training)
        x = x.view(x.size(0), -1)

        x = self.fc1(x, h_net)
        x = F.relu(x)
        if "dropout_fc" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_param, "dropout_fc"), self.training)
        x = self.fc2(x, h_net)
        return x
