from layers.conv2d import *
from layers.linear import *
from utils.dropout_utils import *
from base_model import StnModel

import torch.nn.functional as F
import torch.nn as nn


class StnVgg16(StnModel):
    def __init__(self, num_classes, num_hyper,  h_container):
        super(StnVgg16, self).__init__()
        self.num_hyper = num_hyper
        self.h_container = h_container

        self.conv1 = StnConv2d(3, 64, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv2 = StnConv2d(64, 64, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv3 = StnConv2d(64, 128, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv4 = StnConv2d(128, 128, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv5 = StnConv2d(128, 256, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv6 = StnConv2d(256, 256, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv7 = StnConv2d(256, 256, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv8 = StnConv2d(256, 512, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv9 = StnConv2d(512, 512, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv10 = StnConv2d(512, 512, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv11 = StnConv2d(512, 512, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv12 = StnConv2d(512, 512, kernel_size=3, padding=1, num_hyper=num_hyper)
        self.conv13 = StnConv2d(512, 512, kernel_size=3, padding=1, num_hyper=num_hyper)

        self.fc1 = StnLinear(512, 512, num_hyper, bias=True)
        self.fc2 = StnLinear(512, 512, num_hyper, bias=True)
        self.fc3 = StnLinear(512, num_classes, num_hyper, bias=True)

        self.layers = nn.ModuleList([
            self.conv1, self.conv2, self.conv3,
            self.conv4, self.conv5, self.conv6,
            self.conv7, self.conv8, self.conv9,
            self.conv10, self.conv11, self.conv12,
            self.conv13, self.fc1, self.fc2, self.fc3])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, StnConv2d):
                n = m.kernel_size * m.kernel_size * m.out_channels
                m.general_weight.data.normal_(0, math.sqrt(2. / n))
                m.response_weight.data.normal_(0, math.sqrt(2. / n))
                m.general_bias.data.zero_()
                m.response_bias.data.fill_(0.001)

    def get_layers(self):
        return self.layers

    def forward(self, x, h_net, h_param):
        if "dropout0" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout0"), self.training)

        x = self.conv1(x, h_net)
        x = F.relu(x)
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
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout4" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout4"), self.training)

        x = self.conv5(x, h_net)
        x = F.relu(x)
        if "dropout5" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout5"), self.training)

        x = self.conv6(x, h_net)
        x = F.relu(x)
        if "dropout6" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout6"), self.training)

        x = self.conv7(x, h_net)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout7" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout7"), self.training)

        x = self.conv8(x, h_net)
        x = F.relu(x)
        if "dropout8" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout8"), self.training)

        x = self.conv9(x, h_net)
        x = F.relu(x)
        if "dropout9" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout9"), self.training)

        x = self.conv10(x, h_net)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout10" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout10"), self.training)

        x = self.conv11(x, h_net)
        x = F.relu(x)
        if "dropout11" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout11"), self.training)

        x = self.conv12(x, h_net)
        x = F.relu(x)
        if "dropout12" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout12"), self.training)

        x = self.conv13(x, h_net)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        if "dropout13" in self.h_container.h_dict:
            x = dropout_2d(x, self.h_container.transform_perturbed_hyper(h_param, "dropout13"), self.training)
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
