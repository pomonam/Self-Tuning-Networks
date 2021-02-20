from layers.linear import *
from utils.dropout_utils import dropout
from base_model import StnModel

import torch.nn.functional as F
import torch.nn as nn


class StnThreeLayerMLP(StnModel):
    def __init__(self, input_dim, output_dim, num_hyper, h_container, use_bias=True):
        super(StnThreeLayerMLP, self).__init__()
        self.input_dim = input_dim
        self.layer_structure = [input_dim, 1200, 1200, 1200, output_dim]
        self.num_hyper = num_hyper
        self.h_container = h_container
        self.use_bias = use_bias

        self.layers = nn.ModuleList(
            [StnLinear(self.layer_structure[i], self.layer_structure[i + 1], num_hyper=num_hyper, bias=use_bias)
             for i in range(len(self.layer_structure) - 1)]
        )

    def get_layers(self):
        return self.layers

    def forward(self, x, h_net, h_tensor):
        x = x.view(-1, self.input_dim)
        if "dropout0" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout0"), self.training)

        x = self.layers[0](x, h_net)
        x = F.relu(x)
        if "dropout1" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout1"), self.training)

        x = self.layers[1](x, h_net)
        x = F.relu(x)
        if "dropout2" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout2"), self.training)

        x = self.layers[2](x, h_net)
        return x
