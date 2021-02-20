import torch.nn.functional as F
import torch


def linear(x, low=0., high=0.):
    return x


def linear_inv(x, low=0., high=0.):
    return x


def softplus(x, low=0., high=0.):
    return F.softplus(x)


def softplus_inv(x, low=0., high=0.):
    return torch.log(torch.exp(x) - 1)


def exp(x, low=0., high=0.):
    return torch.exp(x)


def exp_inv(x, low=0., high=0.):
    return torch.log(x)


def logit(x, low=0., high=0.):
    return torch.log(x) - torch.log(1 - x)


def logit_inv(x, low=0., high=0.):
    return sigmoid(x)


def sigmoid(x, low=0., high=0.):
    return torch.sigmoid(x)


def sigmoid_inv(x, low=0., high=0.):
    return logit(x)


def stretch_sigmoid(x, low=0., high=1.):
    return (high - low) * sigmoid(x) + low


def stretch_sigmoid_inv(x, low=0., high=1.):
    return torch.log(x - low) - torch.log(high - x)


def lower_bound_softplus(x, low=0., high=0.):
    return torch.log(1 + torch.exp(x)) + low


def lower_bound_softplus_inv(x, low=0., high=0.):
    return torch.log(torch.exp(x - low) - 1)


def upper_bound_softplus(x, low=0., high=0.):
    return -torch.log(1 + torch.exp(-x + high)) + high


def upper_bound_softplus_inv(x, low=0., high=0.):
    return -torch.log(torch.exp(-x + high) - 1) + high


if __name__ == "__main__":
    # Sanity Check:
    lists = torch.arange(-5, 5).float()
    if torch.all(torch.lt(torch.abs(torch.add(softplus_inv(softplus(lists)), -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(softplus_inv(softplus(lists, low=3, high=5),
                                                           low=3, high=5), -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(exp_inv(exp(lists)), -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(exp_inv(exp(lists, low=3, high=5),
                                                      low=3, high=5), -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(sigmoid_inv(sigmoid(lists)), -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(sigmoid_inv(sigmoid(lists, low=3, high=5),
                                                          low=3, high=5), -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    temp_lists = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).float()
    if torch.all(torch.lt(torch.abs(torch.add(logit_inv(logit(temp_lists)), -temp_lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(logit_inv(logit(temp_lists, low=3, high=5),
                                                        low=3, high=5), -temp_lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(stretch_sigmoid_inv(stretch_sigmoid(lists)),
                                              -lists)), 1e-3)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(stretch_sigmoid_inv(
            stretch_sigmoid(lists, low=3, high=5), low=3, high=5), -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(softplus_inv(softplus(lists)),
                                              -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(softplus_inv(
            softplus(lists, low=3, high=5), low=3, high=5), -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(lower_bound_softplus_inv(lower_bound_softplus(lists)),
                                              -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(lower_bound_softplus_inv(
            lower_bound_softplus(lists, low=3, high=5), low=3, high=5), -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(upper_bound_softplus_inv(upper_bound_softplus(lists)),
                                              -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
    if torch.all(torch.lt(torch.abs(torch.add(upper_bound_softplus_inv(
            upper_bound_softplus(lists, low=3, high=5), low=3, high=5), -lists)), 1e-5)):
        print("success")
    else:
        print("fail")
