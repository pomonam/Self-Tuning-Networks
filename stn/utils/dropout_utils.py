import torch.nn.functional as F
import torch.nn as nn

import torch


def dropout(x, probs, training=False):
    if not training:
        return x
    if isinstance(probs, float):
        return F.dropout(x, probs, training)
    x_size = x.size()
    x = x.view(x.size(0), -1)
    probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
    mask = (1 - probs).bernoulli().div_(1 - probs)
    return (x.float() * mask).view(x_size)


def dropout_2d(x, probs, training=False):
    if not training:
        return x
    if isinstance(probs, float):
        return F.dropout2d(x, probs, training)
    probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
    mask = (1 - probs).bernoulli().div_(1 - probs)
    mask = mask.unsqueeze(2).unsqueeze(2)
    return x * mask


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout):
        if not self.training:
            return x

        if isinstance(dropout, torch.Tensor):
            m = x.data.new(1, x.size(1), x.size(2))
            expanded_dropout_probs = (1 - dropout.unsqueeze(0).expand_as(m)).detach()
            mask = expanded_dropout_probs.bernoulli() / expanded_dropout_probs
        else:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = m / (1 - dropout)

        mask = mask.expand_as(x)
        return mask * x
