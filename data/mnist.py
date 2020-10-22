from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets

import numpy as np
import torch


class StnMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(StnMNIST, self).__init__(*args, **kwargs)

    def set_h_container(self, h_container, perturbed_h_tensor):
        pass

    def reset_hyper_params(self):
        pass


def mnist_mlp_loader(info, root_dir="data/"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = StnMNIST(root=root_dir, train=True, download=True, transform=transform)
    num_train = int(np.floor((1 - info["percent_valid"]) * len(train_set)))

    val_set = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)

    train_set.data = train_set.data[:num_train, :, :]
    train_set.targets = train_set.targets[:num_train]

    val_set.data = val_set.data[num_train:, :, :]
    val_set.targets = val_set.targets[num_train:]

    test_set = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=info["train_batch_size"],
                                               shuffle=True,
                                               num_workers=0,
                                               worker_init_fn=np.random.seed(info["data_seed"]))
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=info["valid_batch_size"],
                                             shuffle=True,
                                             num_workers=0,
                                             worker_init_fn=np.random.seed(info["data_seed"]))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=info["test_batch_size"],
                                              shuffle=True,
                                              num_workers=0,
                                              worker_init_fn=np.random.seed(info["data_seed"]))

    return train_loader, val_loader, test_loader
