from torchvision import datasets
from stn.utils.cutout_utils import Cutout
from torchvision import transforms
from PIL import Image

import numpy as np

import torch


class StnCIFAR10(datasets.CIFAR10):
    """ CIFAR 10 dataset."""
    def __init__(self, tune_affine, tune_color, tune_cutout, *args, **kwargs):
        super(StnCIFAR10, self).__init__(*args, **kwargs)

        self.h_container = None
        self.perturbed_h_tensor = 0
        self.tune_affine = tune_affine
        self.tune_color = tune_color
        self.tune_cutout = tune_cutout
        self.affine_transform = transforms.Compose([])
        self.h_flip_transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        self.jitter_transform = transforms.Compose([])
        self.cut_transform = Cutout(n_holes=-1, length=-1)
        self.num_processed = 0

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        img = self.h_flip_transform(img)
        perturbed_img = img.copy()

        if self.tune_affine:
            self.set_affine()
            if self.train:
                img = self.affine_transform(img)
                self.set_perturbed_affine()
                perturbed_img = self.affine_transform(perturbed_img)

        if self.tune_color:
            self.set_jitters()
            if self.train:
                img = self.jitter_transform(img)
                self.set_perturbed_jitters()
                perturbed_img = self.jitter_transform(perturbed_img)

        if self.transform is not None:
            img = self.transform(img)
            perturbed_img = self.transform(perturbed_img)

        if self.tune_cutout:
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
        if self.tune_cutout:
            self.reg_dict["cutout_length"] = self.h_container.transform("cutout_length").item()
            self.reg_dict["cutout_holes"] = self.h_container.transform("cutout_holes").item()
            self.aug_dict["cutout_length"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                        "cutout_length")
            self.aug_dict["cutout_holes"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                        "cutout_holes")
        if self.tune_color:
            self.reg_dict["brightness"] = self.h_container.transform("brightness").item()
            self.reg_dict["contrast"] = self.h_container.transform("contrast").item()
            self.reg_dict["saturation"] = self.h_container.transform("saturation").item()
            self.reg_dict["hue"] = self.h_container.transform("hue").item()
            self.aug_dict["brightness"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                     "brightness")
            self.aug_dict["contrast"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                   "contrast")
            self.aug_dict["saturation"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                     "saturation")
            self.aug_dict["hue"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                              "hue")
        if self.tune_affine:
            self.reg_dict["degrees"] = self.h_container.transform("degrees").item()
            self.reg_dict["translate"] = self.h_container.transform("translate").item()
            self.reg_dict["shear"] = self.h_container.transform("shear").item()
            self.reg_dict["scale"] = self.h_container.transform("scale").item()
            self.aug_dict["degrees"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                  "degrees")
            self.aug_dict["translate"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                    "translate")
            self.aug_dict["shear"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                "shear")
            self.aug_dict["scale"] = self.h_container.transform_perturbed_hyper(self.perturbed_h_tensor,
                                                                                "scale")

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

    def set_jitters(self):
        if self.h_container is None:
            self.jitter_transform = transforms.Compose([])
            return
        self.jitter_transform = transforms.ColorJitter(brightness=self.reg_dict["brightness"],
                                                       contrast=self.reg_dict["contrast"],
                                                       saturation=self.reg_dict["saturation"],
                                                       hue=self.reg_dict["hue"])

    def set_perturbed_jitters(self):
        if self.h_container is None:
            self.jitter_transform = transforms.Compose([])
            return
        if self.perturbed_h_tensor is None:
            self.set_jitters()
            return
        self.jitter_transform = transforms.ColorJitter(brightness=self.aug_dict["brightness"][self.num_processed].item(),
                                                       contrast=self.aug_dict["contrast"][self.num_processed].item(),
                                                       saturation=self.aug_dict["saturation"][self.num_processed].item(),
                                                       hue=self.aug_dict["hue"][self.num_processed].item())

    def set_affine(self):
        if self.h_container is None:
            self.affine_transform = transforms.Compose([])
            return
        self.affine_transform = transforms.RandomAffine(degrees=self.reg_dict["degrees"],
                                                        translate=(self.reg_dict["translate"],
                                                                   self.reg_dict["translate"]),
                                                        scale=(1 - self.reg_dict["scale"],
                                                               1 + self.reg_dict["scale"]),
                                                        shear=self.reg_dict["shear"])

    def set_perturbed_affine(self):
        if self.h_container is None:
            self.affine_transform = transforms.Compose([])
            return
        if self.perturbed_h_tensor is None:
            self.set_affine()
            return
        self.affine_transform = transforms.RandomAffine(degrees=self.aug_dict["degrees"][self.num_processed].item(),
                                                        translate=(self.aug_dict["translate"][self.num_processed].item(),
                                                                   self.aug_dict["translate"][self.num_processed].item()),
                                                        scale=(1 - self.aug_dict["scale"][self.num_processed].item(),
                                                               1 + self.aug_dict["scale"][self.num_processed].item()),
                                                        shear=self.aug_dict["shear"][self.num_processed].item())

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


def stn_cifar10_loader(info, root_dir='data/'):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_set = StnCIFAR10(root=root_dir,
                           tune_affine=info["tune_affine"],
                           tune_color=info["tune_color_jitter"],
                           tune_cutout=info["tune_cutout"],
                           download=True, train=True, transform=train_transform)
    num_train = int(np.floor((1 - info["percent_valid"]) * len(train_set)))

    train_set.data = train_set.data[:num_train, :, :]
    train_set.targets = train_set.targets[:num_train]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=info["train_batch_size"],
                                               shuffle=True, num_workers=0,
                                               worker_init_fn=np.random.seed(info["data_seed"]))

    val_set = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=test_transform)
    val_set.data = val_set.data[num_train:, :, :]
    val_set.targets = val_set.targets[num_train:]

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=info["valid_batch_size"],
                                             shuffle=True, num_workers=0,
                                             worker_init_fn=np.random.seed(info["data_seed"]))

    test_set = datasets.CIFAR10(root_dir, download=True, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=info["test_batch_size"],
                                              shuffle=True, num_workers=0,
                                              worker_init_fn=np.random.seed(info["data_seed"]))

    return train_loader, val_loader, test_loader
