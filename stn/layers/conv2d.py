from torch.nn import init

import torch.nn.functional as F
import torch.nn as nn

import math
import torch


class StnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_hyper,
                 stride=1, padding=0, groups=1, dilation=1, bias=True):
        """ Initialize a class StnConv2d.
        :param in_channels: int
        :param out_channels: int
        :param kernel_size: int or (int, int)
        :param num_hyper: int
        :param stride: int or (int, int)
        :param padding: int or (int, int)
        :param groups: int
        :param dilation: int or (int, int)
        :param bias: bool
        """
        super(StnConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_hyper = num_hyper
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.general_weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        self.response_weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))

        self.general_parameters = [self.general_weight]
        self.response_parameters = [self.response_weight]

        if bias:
            self.general_bias = nn.Parameter(torch.Tensor(out_channels))
            self.general_parameters.append(self.general_bias)
            self.response_bias = nn.Parameter(torch.Tensor(out_channels))
            self.response_parameters.append(self.response_bias)
        else:
            self.register_parameter("general_bias", None)
            self.register_parameter("response_bias", None)

        self.hyper_bottleneck = nn.Linear(
            self.num_hyper, self.out_channels * 2, bias=False)
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
        :param inputs: Tensor of size 'batch_size x in_channels x height x width'
        :param h_net: Tensor of size 'batch_size x num_hyper'
        :return: Tensor of size 'batch_size x out_channels x height x width'
        """
        output = F.conv2d(inputs, self.general_weight, self.general_bias, padding=self.padding,
                          stride=self.stride, groups=self.groups, dilation=self.dilation)

        linear_hyper = self.hyper_bottleneck(h_net)
        hyper_weight = linear_hyper[:, :self.out_channels].unsqueeze(2).unsqueeze(2)
        hyper_bias = linear_hyper[:, self.out_channels:]
        response_out = F.conv2d(inputs, self.response_weight, padding=self.padding, stride=self.stride,
                                groups=self.groups, dilation=self.dilation)
        response_out *= hyper_weight

        if self.response_bias is not None:
            response_out += (hyper_bias * self.response_bias).unsqueeze(2).unsqueeze(2)
        output += response_out

        return output

    def extra_repr(self):
        """ Representation for StnConv2d.
        :return: str
        """
        s = ("{in_channels}, {out_channels}, kernel_size={kernel_size}"
             ", num_hyper={num_hyper}, stride={stride}")
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.general_bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)
