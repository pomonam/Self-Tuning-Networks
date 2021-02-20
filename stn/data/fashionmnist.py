from torchvision import datasets
from stn.utils.cutout_utils import Cutout
from torchvision import transforms

import numpy as np

import copy
import torch


class StnFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST dataset."""
    def __init__(self, *args, **kwargs):
        super(StnFashionMNIST, self).__init__(*args, **kwargs)
        self.h_container = None
        self.cut_transform = Cutout(n_holes=-1, length=-1)
        self.num_processed = 0

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        perturbed_img = copy.deepcopy(img)

        self.set_cutout()
        if self.train:
            img = self.cut_transform(img)
            self.set_perturbed_cutout()
            perturbed_img = self.cut_transform(perturbed_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        self.num_processed += 1
        return img, target, perturbed_img

    def cache_dir(self):
        self.reg_dict = {}
        self.aug_dict = {}
        self.reg_dict["cutout_length"] = self.h_container.transform("cutout_length").item()
        self.reg_dict["cutout_holes"] = self.h_container.transform("cutout_holes").item()
        self.aug_dict["cutout_length"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                    "cutout_length")
        self.aug_dict["cutout_holes"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                   "cutout_holes")

    def set_h_container(self, h_container, perturbed_h_tensor):
        self.num_processed = 0
        self.h_container = h_container
        self.perturbed_h_tensor = perturbed_h_tensor
        # Cache for efficiency.
        self.cache_dir()

    def reset_hyper_params(self):
        self.h_container = None
        self.perturbed_h_tensor = None
        self.reg_dict = {}
        self.aug_dict = {}

    def set_cutout(self):
        if self.h_container is None:
            self.cut_transform = Cutout(n_holes=-1, length=-1)
            return
        if "cutout_length" in self.h_container.h_dict:
            self.cut_transform.length = int(self.reg_dict["cutout_length"])
        if "cutout_holes" in self.h_container.h_dict:
            self.cut_transform.n_holes = int(self.reg_dict["cutout_holes"])

    def set_perturbed_cutout(self):
        if self.h_container is None:
            self.cut_transform = Cutout(n_holes=-1, length=-1)
            return
        if self.perturbed_h_tensor is None:
            self.set_cutout()
            return
        if "cutout_length" in self.h_container.h_dict:
            self.cut_transform.length = int(self.aug_dict["cutout_length"][self.num_processed].item())
        if "cutout_holes" in self.h_container.h_dict:
            self.cut_transform.n_holes = int(self.aug_dict["cutout_holes"][self.num_processed].item())


def stn_fashion_mnist_loader(info, root_dir='data/'):
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data
    train_set = StnFashionMNIST(root_dir, download=True, train=True, transform=transform)
    num_train = int(np.floor((1 - info["percent_valid"]) * len(train_set)))

    train_set.data = train_set.data[:num_train, :, :]
    train_set.targets = train_set.targets[:num_train]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=info["train_batch_size"], shuffle=True,
                                               worker_init_fn=np.random.seed(info["data_seed"]))

    val_set = datasets.FashionMNIST(root=root_dir, train=True, download=True, transform=transform)
    val_set.data = val_set.data[num_train:, :, :]
    val_set.targets = val_set.targets[num_train:]

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=info["valid_batch_size"], shuffle=True,
                                             worker_init_fn=np.random.seed(info["data_seed"]))

    test_set = datasets.FashionMNIST(root_dir, download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=info["test_batch_size"], shuffle=True,
                                              worker_init_fn=np.random.seed(info["data_seed"]))

    return train_loader, val_loader, test_loader
