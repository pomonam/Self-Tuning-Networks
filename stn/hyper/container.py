from collections import OrderedDict
from stn.hyper.transformation import *

import torch.nn.functional as F
import torch.nn as nn
import torch
import math


class HyperInfo:
    """ A class that stores information on each hyperparameter. """
    def __init__(self, name, index, domain, transform_style, discrete=False, same_perturb_mb=False):
        """ Initialize a class HyperInfo.
        :param name: str
        :param index: int
        :param domain: (float, float) or (str, float) or (str)
        :param transform_style: str
        :param discrete: bool
        :param same_perturb_mb: bool
        """
        self.name = name
        self.index = index
        self.domain = domain
        self.transform_style = transform_style
        self.discrete = discrete
        self.same_perturb_mb = same_perturb_mb


class HyperContainer:
    """ A class that stores all hyperparameters. """

    _TRANSFORMATION_MAP = {
        "softplus": softplus,
        "exp": exp,
        "s_sigmoid": stretch_sigmoid,
        "upper_softplus": upper_bound_softplus,
        "lower_softplus": lower_bound_softplus,
        "logit": logit,
        "sigmoid": sigmoid,
        "linear": linear
    }

    _INV_TRANSFORMATION_MAP = {
        "softplus": softplus_inv,
        "exp": exp_inv,
        "s_sigmoid": stretch_sigmoid_inv,
        "upper_softplus": upper_bound_softplus_inv,
        "lower_softplus": lower_bound_softplus_inv,
        "logit": logit_inv,
        "sigmoid": sigmoid_inv,
        "linear": linear_inv
    }

    def __init__(self, device):
        """ Initialize a class HyperContainer.
        :param device: Device
        """
        self.h_tensor_lst = []
        self.h_scale_lst = []

        self.h_tensor = None
        self.h_scale = None

        self.h_dict = OrderedDict()
        self.device = device

    def register(self, name, value, scale, min_range, max_range, discrete=False,
                 same_perturb_mb=False, manual_trans=None):
        """ Registers the hyperparameter to tune.
        :param name: str
        :param value: float
        :param scale: float
        :param min_range: float or int
        :param max_range: float or int
        :param discrete: bool
        :param same_perturb_mb: bool
        :param manual_trans: function
        :return: None
        """
        if manual_trans is not None:
            if manual_trans not in self._TRANSFORMATION_MAP.keys():
                raise Exception("Manual transformation {} not available.".format(str(manual_trans)))
            trans = manual_trans
        elif max_range != "inf" and min_range != "inf":
            trans = "s_sigmoid"
        elif max_range != "inf" and min_range == "inf":
            trans = "upper_softplus"
        elif max_range == "inf" and min_range != "inf":
            trans = "lower_softplus"
        elif max_range == "inf" and min_range == "inf":
            trans = "linear"
        else:
            raise AssertionError()

        self.h_tensor_lst.append(self._INV_TRANSFORMATION_MAP[trans](torch.tensor(value), min_range, max_range))
        self.h_scale_lst.append(softplus_inv(torch.tensor(scale)))

        self.h_dict[name] = HyperInfo(name=name,
                                      index=len(self.h_dict),
                                      domain=(min_range, max_range),
                                      transform_style=trans,
                                      discrete=discrete,
                                      same_perturb_mb=same_perturb_mb)

        self.h_tensor = nn.Parameter(torch.stack(self.h_tensor_lst).to(self.device),
                                     requires_grad=True)
        self.h_scale = nn.Parameter(torch.stack(self.h_scale_lst).to(self.device),
                                    requires_grad=True)

    def get_size(self):
        """ Return the number of hyperparameter tuning.
        :return: int
        """
        if self.h_tensor is None:
            raise Exception("Please register at least one hyperparameter.")
        return self.h_tensor.size(0)

    def get_labels(self, tune_scales=True):
        """ Return the labels of hyperparameters.
        :return: tuple(str)
        """
        h_labels = list(self.h_dict.keys())
        if tune_scales:
            h_labels += [h_label + "_scale" for h_label in h_labels]
        h_labels = tuple(h_labels)
        return h_labels

    def transform(self, name):
        """ Transform the hyperparameter given its name.
        Note that this function returns the non-perturbed hyperparameter.
        :param name: str
        :return: float
        """
        min_range = self.h_dict[name].domain[0]
        max_range = self.h_dict[name].domain[1]
        hyper_value = self.h_tensor[self.h_dict[name].index]
        return self._TRANSFORMATION_MAP[self.h_dict[name].transform_style](hyper_value, min_range, max_range)

    def get_perturbed_hyper(self, batch_size, same_perturb_mb=False):
        """ Returns a perturbed set of hyperparameter given the batch size.
        :param batch_size: int
        :param same_perturb_mb: bool
        :return: Tensor of size 'batch_size x num_hyper'
        """
        noise = self.h_tensor.new(batch_size, self.get_size()).normal_()
        perturb_h_tensor = self.h_tensor + F.softplus(self.h_scale) * noise

        if same_perturb_mb:
            perturb_h_tensor = perturb_h_tensor[0].repeat((batch_size, 1))
        else:
            for h_info in [h_info for h_info in self.h_dict.values()]:
                h_idx = h_info.index
                if h_info.same_perturb_mb:
                    # If per mini-batch, just repeat them.
                    perturb_h_tensor[:, h_idx] = perturb_h_tensor[0, h_idx].repeat((batch_size,))
        return perturb_h_tensor

    def transform_perturbed_hyper(self, h_tensor, name):
        """ Given h_tensor (either perturbed or non perturbed), return the transformed h_tensor.
        :param h_tensor: Tensor of size 'batch_size x 1'
        :param name: str
        :return: Tensor of size 'batch_size x 1'
        """
        min_range = self.h_dict[name].domain[0]
        max_range = self.h_dict[name].domain[1]

        transformed = self._TRANSFORMATION_MAP[self.h_dict[name].transform_style](
            h_tensor[:, self.h_dict[name].index], min_range, max_range)
        if self.h_dict[name].discrete:
            transformed = torch.floor(transformed)
        return transformed

    def get_entropy(self):
        """ Return the entropy of the hyperparameter distribution.
        :return: float
        """
        scale = F.softplus(self.h_scale)
        return torch.sum(torch.log(scale * math.sqrt(2 * math.pi * math.e)))

    def __str__(self):
        """ String representation of HyperContainer.
        :return: str
        """
        return str(self.generate_summary(show_scale=False))

    def generate_summary(self, show_scale=True):
        """ Return a dictionary containing values of all hyperparameters.
        :param show_scale bool
        :return: dict
        """
        h_stats = OrderedDict()
        for h_name, h_info in self.h_dict.items():
            h_stats[h_name] = self.transform(h_name)
            if show_scale:
                h_stats[h_name + '_scale'] = F.softplus(self.h_scale[h_info.index])
        return h_stats

    def generate_perturbed_summary(self, perturbed_h_tensor):
        """ Return a dictionary containing values of perturbed_h_tensor hyperparameters.
        :param perturbed_h_tensor Tensor of size "num_hyper"
        :return: dict
        """
        h_stats = OrderedDict()
        for h_name, h_info in self.h_dict.items():
            if "dropout" not in h_name:
                min_range = self.h_dict[h_name].domain[0]
                max_range = self.h_dict[h_name].domain[1]
                h_stats[h_name] = self._TRANSFORMATION_MAP[self.h_dict[h_name].transform_style](
                    perturbed_h_tensor[self.h_dict[h_name].index], min_range, max_range).item()
        return h_stats
