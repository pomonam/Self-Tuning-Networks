from abc import ABCMeta
from layers.linear import *
from layers.conv2d import *

import torch.nn as nn


# Add custom layers here.
_STN_LAYERS = [StnLinear, StnConv2d]


class StnModel(nn.Module, metaclass=ABCMeta):
    # Initialize an attribute self.layers (a list containing all layers).
    def get_layers(self):
        raise NotImplementedError

    def get_response_parameters(self):
        """ Return the response parameters.
        :return: List[Tensors]
        """
        params = []
        for idx, layer in enumerate(self.get_layers()):
            for stn_layer in _STN_LAYERS:
                if isinstance(layer, stn_layer):
                    params = params + layer.response_parameters
        return params

    def get_general_parameters(self):
        """ Return the general parameters.
        :return: List[Tensors]
        """
        params = []
        for idx, layer in enumerate(self.get_layers()):
            is_stn_layer = False
            for stn_layer in _STN_LAYERS:
                if isinstance(layer, stn_layer):
                    is_stn_layer = True
                    params = params + layer.general_parameters
                    break
            if not is_stn_layer:
                params = params + [p for p in layer.parameters()]
        return params

    def forward(self, x, h_net, h_param):
        """ A forward pass for StnModel.
        :param x: Input Tensor
        :param h_net: Tensor of size 'batch_size x num_hyper'
        :param h_param: Tensor of size 'batch_size x num_hyper'
        :return: Output Tensor
        """
        raise NotImplementedError()
