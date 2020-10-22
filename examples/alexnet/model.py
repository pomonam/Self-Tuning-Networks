from layers.conv2d import *
from layers.linear import *
from utils.dropout_utils import *
from base_model import StnModel

import torch.nn.functional as F
import torch.nn as nn


class StnAlexNet(StnModel):
    def __init__(self, num_classes, num_hyper,  h_container):
        super(StnAlexNet, self).__init__()
        self.num_hyper = num_hyper
        self.h_container = h_container
        self.filters = [3, 64, 192, 384, 256, 256]

        self.conv1 = StnConv2d(3, 64, stride=2, kernel_size=3, padding=1,
                               num_hyper=num_hyper, bias=True)
        self.conv2 = StnConv2d(64, 192, stride=1, kernel_size=3, padding=1,
                               num_hyper=num_hyper, bias=True)
        self.conv3 = StnConv2d(192, 384, stride=1, kernel_size=3, padding=1,
                               num_hyper=num_hyper, bias=True)
        self.conv4 = StnConv2d(384, 256, stride=1, kernel_size=3, padding=1,
                               num_hyper=num_hyper, bias=True)
        self.conv5 = StnConv2d(256, 256, stride=1, kernel_size=3, padding=1,
                               num_hyper=num_hyper, bias=True)

        self.last_dim = 256 * 2 * 2
        self.fc1 = StnLinear(self.last_dim, 4096, num_hyper=num_hyper, bias=True)
        self.fc2 = StnLinear(4096, 4096, num_hyper=num_hyper, bias=True)
        self.fc3 = StnLinear(4096, num_classes, num_hyper=num_hyper, bias=True)

        self.layers = nn.ModuleList([
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
            self.fc1, self.fc2, self.fc3
        ])

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

        x = self.conv3(x, h_net)
        x = F.relu(x)
        if "dropout3" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout3"), self.training)

        x = self.conv4(x, h_net)
        x = F.relu(x)
        if "dropout4" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout4"), self.training)

        x = self.conv5(x, h_net)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout5" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout5"), self.training)
        x = x.view(x.size(0), -1)

        x = self.fc1(x, h_net)
        x = F.relu(x)
        if "dropout_fc1" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_param, "dropout_fc1"), self.training)

        x = self.fc2(x, h_net)
        x = F.relu(x)
        if "dropout_fc2" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_param, "dropout_fc2"), self.training)

        x = self.fc3(x, h_net)
        return x
