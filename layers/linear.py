from torch.nn import init

import torch.nn.functional as F
import torch.nn as nn

import torch
import math


class StnLinear(nn.Module):
    def __init__(self, in_features, out_features, num_hyper, bias=True):
        """ Initialize a class StnLinear.
        :param in_features: int
        :param out_features: int
        :param num_hyper: int
        :param bias: bool
        """
        super(StnLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_hyper = num_hyper

        self.general_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.response_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.general_parameters = [self.general_weight]
        self.response_parameters = [self.response_weight]

        if bias:
            self.general_bias = nn.Parameter(torch.Tensor(out_features))
            self.general_parameters.append(self.general_bias)
            self.response_bias = nn.Parameter(torch.Tensor(out_features))
            self.response_parameters.append(self.response_bias)
        else:
            self.register_parameter("general_bias", None)
            self.register_parameter("response_bias", None)

        self.hyper_bottleneck = nn.Linear(num_hyper, out_features * 2, bias=False)
        self.response_parameters.append(self.hyper_bottleneck.weight)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights and bias.
        :return: None
        """
        init.kaiming_uniform_(self.general_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.response_weight, a=math.sqrt(5))
        if self.general_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.general_weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.general_bias, -bound, bound)
            init.uniform_(self.response_bias, -bound, bound)
        self.hyper_bottleneck.weight.data.fill_(0)

    def forward(self, inputs, h_net):
        """ Returns a forward pass.
        :param inputs: Tensor of size 'batch_size x in_features'
        :param h_net: Tensor of size 'batch_size x num_hyper'
        :return: Tensor of size 'batch_size x out_features'
        """
        output = F.linear(inputs, self.general_weight, self.general_bias)

        linear_hyper = self.hyper_bottleneck(h_net)
        hyper_weight = linear_hyper[:, :self.out_features]
        hyper_bias = linear_hyper[:, self.out_features:]
        response_out = hyper_weight * F.linear(inputs, self.response_weight)

        if self.response_bias is not None:
            response_out += hyper_bias * self.response_bias
        output += response_out
        return output

    def extra_repr(self):
        """ Representation for StnLinear.
        :return: str
        """
        return "in_features={}, out_features={}, num_hyper={}, bias={}".format(
            self.in_features, self.out_features, self.num_hyper, self.general_bias is not None)
