from data.cifar10 import *
from examples.vgg.model import *
from hyper.container import HyperContainer
from base_step_optimizer import *
from base_trainer import *

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import argparse
import sys
import random
import torch
import wandb
import os


parser = argparse.ArgumentParser(description="VGG STN Experiment")
parser.add_argument("--experiment_name", type=str, default="vgg_stn")

parser.add_argument("--delta_stn", action="store_true", default=False)
parser.add_argument("--linearize", action="store_true", default=False)
parser.add_argument("--decay_both", action="store_true", default=True)

# Tuning options:
parser.add_argument("--tune_scales", action="store_true", default=False)

parser.add_argument("--tune_dropout", action="store_true", default=True)
parser.add_argument("--initial_dropout_value", type=float, default=0.05)
parser.add_argument("--initial_dropout_scale", type=float, default=1.)

parser.add_argument("--tune_cutout", action="store_true", default=True)
parser.add_argument("--initial_cutout_num", type=int, default=1.)
parser.add_argument("--initial_cutout_length", type=int, default=4.)
parser.add_argument("--initial_cutout_scale", type=float, default=1.)

parser.add_argument("--tune_affine", action="store_true", default=True)
parser.add_argument("--initial_degrees", type=float, default=5.)
parser.add_argument("--initial_degrees_scale", type=float, default=1.)
parser.add_argument("--initial_translate", type=float, default=0.05)
parser.add_argument("--initial_translate_scale", type=float, default=1.)
parser.add_argument("--initial_shear", type=float, default=5.)
parser.add_argument("--initial_shear_scale", type=float, default=1.)
parser.add_argument("--initial_scale", type=float, default=0.05)
parser.add_argument("--initial_scale_scale", type=float, default=1.)

parser.add_argument("--tune_color_jitter", action="store_true", default=True)
parser.add_argument("--initial_brightness", type=float, default=0.05)
parser.add_argument("--initial_brightness_scale", type=float, default=1.0)
parser.add_argument("--initial_sat", type=float, default=0.05)
parser.add_argument("--initial_sat_scale", type=float, default=1.0)
parser.add_argument("--initial_contrast",  type=float, default=0.05)
parser.add_argument("--initial_contrast_scale", type=float, default=1.0)
parser.add_argument("--initial_hue", type=float, default=0.05)
parser.add_argument("--initial_hue_scale", type=float, default=1.0)

parser.add_argument("--percent_valid", type=float, default=0.20)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--valid_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=500)
parser.add_argument("--total_epochs", type=int, default=200)
parser.add_argument("--warmup_epochs", type=int, default=5)

parser.add_argument("--train_lr", type=float, default=3e-2)
parser.add_argument("--valid_lr", type=float, default=1e-2)
parser.add_argument("--scale_lr", type=float, default=None)

parser.add_argument("--optimizer", type=str, default="rmsprop",
                    choices=["adam", "rmsprop"])
parser.add_argument("--train_steps", type=int, default=5)
parser.add_argument("--valid_steps", type=int, default=1)
parser.add_argument("--entropy_weight", type=float, default=1e-3)
parser.add_argument("--early_stop_patience", type=int, default=None)
parser.add_argument("--lr_decay_patience", type=int, default=None)
parser.add_argument("--lr_decay", type=float, default=0.2)

parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--no_cuda", action="store_true", default=False)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--data_seed", type=int, default=0)
parser.add_argument("--model_seed", type=int, default=0)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
cudnn.benchmark = True
cudnn.deterministic = True

torch.manual_seed(args.data_seed)
np.random.seed(args.data_seed)
random.seed(args.data_seed)
if args.cuda:
    torch.cuda.manual_seed(args.data_seed)
    torch.cuda.manual_seed_all(args.data_seed)

wandb.init(project=args.experiment_name,
           tensorboard=True,
           dir=os.getcwd() if args.save_dir is None else args.save_dir)
wandb.config.update(args)

info = vars(args)

# Load data.
train_loader, valid_loader, test_loader = stn_cifar10_loader(info)


# Configure hyperparameters.
h_container = HyperContainer(device)

if info["tune_dropout"]:
    for i in range(14):
        h_container.register(name="dropout{}".format(i),
                             value=info["initial_dropout_value"],
                             scale=info["initial_dropout_scale"],
                             min_range=0., max_range=0.95,
                             discrete=False, same_perturb_mb=False)

    h_container.register("dropout_fc1",
                         info["initial_dropout_value"],
                         info["initial_dropout_scale"],
                         min_range=0., max_range=0.95,
                         discrete=False, same_perturb_mb=False)

    h_container.register("dropout_fc2",
                         info["initial_dropout_value"],
                         info["initial_dropout_scale"],
                         min_range=0., max_range=0.95,
                         discrete=False, same_perturb_mb=False)

if info["tune_cutout"]:
    h_container.register("cutout_length",
                         info["initial_cutout_length"],
                         info["initial_cutout_scale"],
                         min_range=2., max_range=24.,
                         discrete=True, same_perturb_mb=False)

    h_container.register("cutout_holes",
                         info["initial_cutout_num"],
                         info["initial_cutout_scale"],
                         min_range=0., max_range=5.,
                         discrete=True, same_perturb_mb=False)


if info["tune_color_jitter"]:
    h_container.register("hue",
                         info["initial_hue"],
                         info["initial_hue_scale"],
                         min_range=0., max_range=0.5,
                         discrete=False, same_perturb_mb=False)

    h_container.register("brightness",
                         info["initial_brightness"],
                         info["initial_brightness_scale"],
                         min_range=0., max_range=1.,
                         discrete=False, same_perturb_mb=False)

    h_container.register("contrast",
                         info["initial_contrast"],
                         info["initial_contrast_scale"],
                         min_range=0., max_range=1.,
                         discrete=False, same_perturb_mb=False)

    h_container.register("saturation",
                         info["initial_sat"],
                         info["initial_sat_scale"],
                         min_range=0., max_range=1.,
                         discrete=False, same_perturb_mb=False)

if info["tune_affine"]:
    h_container.register("degrees",
                         info["initial_degrees"],
                         info["initial_degrees_scale"],
                         min_range=0., max_range=45.,
                         discrete=False, same_perturb_mb=False)

    h_container.register("shear",
                         info["initial_shear"],
                         info["initial_shear_scale"],
                         min_range=0., max_range=45.,
                         discrete=False, same_perturb_mb=False)

    h_container.register("scale",
                         info["initial_scale"],
                         info["initial_scale_scale"],
                         min_range=0., max_range=0.5,
                         discrete=False, same_perturb_mb=False)

    h_container.register("translate",
                         info["initial_translate"],
                         info["initial_translate_scale"],
                         min_range=0., max_range=0.5,
                         discrete=False, same_perturb_mb=False)

num_hyper = h_container.get_size()

# Define models and optimizers.
torch.manual_seed(args.model_seed)
np.random.seed(args.model_seed)
random.seed(args.model_seed)
if args.cuda:
    torch.cuda.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)

model = StnVgg16(h_container=h_container, num_hyper=num_hyper, num_classes=10)
model = model.to(device)
criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

total_params = sum(param.numel() for param in model.parameters())
print("Args:", args)
print("Model total parameters:", total_params)

if info["delta_stn"]:
    model_general_optimizer = torch.optim.SGD(model.get_general_parameters(),
                                              lr=args.train_lr,
                                              momentum=0.9)
    model_response_optimizer = torch.optim.SGD(model.get_response_parameters(),
                                               lr=args.train_lr,
                                               momentum=0.9)

    if info["optimizer"] == "adam":
        hyper_optimizer = torch.optim.Adam([h_container.h_tensor], lr=args.valid_lr)
        scale_optimizer = torch.optim.Adam([h_container.h_scale], lr=args.valid_lr)
    else:
        hyper_optimizer = torch.optim.RMSprop([h_container.h_tensor], lr=args.valid_lr)
        scale_optimizer = torch.optim.RMSprop([h_container.h_scale],
                                              lr=args.scale_lr if args.scale_lr is not None else args.valid_lr)

    # lr_scheduler1 = ReduceLROnPlateau(model_general_optimizer, "min", patience=info["lr_decay_patience"],
    #                                   factor=info["lr_decay"], verbose=True)
    # lr_scheduler2 = ReduceLROnPlateau(model_response_optimizer, "min", patience=info["lr_decay_patience"],
    #                                   factor=info["lr_decay"], verbose=True)
    if info["decay_both"]:
        lr_scheduler1 = MultiStepLR(model_general_optimizer, milestones=[60, 120, 180], gamma=info["lr_decay"])
        lr_scheduler2 = MultiStepLR(model_response_optimizer, milestones=[60, 120, 180], gamma=info["lr_decay"])
        lr_scheduler = [lr_scheduler1, lr_scheduler2]
    else:
        lr_scheduler = MultiStepLR(model_general_optimizer, milestones=[60, 120, 180], gamma=info["lr_decay"])

    stn_step_optimizer = DeltaStnStepOptimizer(model, model_general_optimizer, model_response_optimizer,
                                               hyper_optimizer, scale_optimizer, criterion, h_container,
                                               info["tune_scales"], info["entropy_weight"], info["linearize"])
else:
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr, momentum=0.9)

    if info["optimizer"] == "adam":
        hyper_optimizer = torch.optim.Adam([h_container.h_tensor], lr=args.valid_lr)
        scale_optimizer = torch.optim.Adam([h_container.h_scale], lr=args.valid_lr)
    else:
        hyper_optimizer = torch.optim.RMSprop([h_container.h_tensor], lr=args.valid_lr)
        scale_optimizer = torch.optim.RMSprop([h_container.h_scale],
                                              lr=args.scale_lr if args.scale_lr is not None else args.valid_lr)

    # lr_scheduler = ReduceLROnPlateau(model_optimizer, "min", patience=info["lr_decay_patience"],
    #                                  factor=info["lr_decay"], verbose=True)
    lr_scheduler = MultiStepLR(model_optimizer, milestones=[60, 120, 180], gamma=info["lr_decay"])

    stn_step_optimizer = StnStepOptimizer(model, model_optimizer, hyper_optimizer, scale_optimizer, criterion,
                                          h_container, info["tune_scales"], info["entropy_weight"])

# Evaluation functions.
def delta_stn_per_epoch_evaluate(current_epoch, train_loss=None):
    def evaluate(loader):
        model.eval()
        correct = total = loss = 0.
        with torch.no_grad():
            for data in loader:
                images, labels = data[0].to(device), data[1].to(device)
                repeated_h_tensor = h_container.h_tensor.unsqueeze(0).repeat((images.shape[0], 1))
                pred = model(images, repeated_h_tensor - repeated_h_tensor.detach(), repeated_h_tensor)
                loss += F.cross_entropy(pred.float(), labels.long(), reduction="sum").item()
                hard_pred = torch.max(pred, 1)[1]
                total += labels.size(0)
                correct += (hard_pred == labels).sum().item()
        accuracy = correct / float(total)
        mean_loss = loss / float(total)
        return mean_loss, accuracy

    train_loader.dataset.reset_hyper_params()
    if train_loss is None:
        train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(valid_loader)
    tst_loss, tst_acc = evaluate(test_loader)

    print("=" * 80)
    print("Train Epoch: {} | Trn Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}"
          " | Test Loss: {:.3f} | Test Acc: {:.3f}".format(current_epoch, train_loss, val_loss,
                                                           val_acc, tst_loss, tst_acc))
    print("=" * 80)

    epoch_dict = {"epoch": current_epoch,
                  "train_loss": train_loss,
                  "val_loss": val_loss,
                  "val_acc": val_acc,
                  "test_loss": tst_loss,
                  "test_acc": tst_acc,
                  "glr": model_general_optimizer.param_groups[0]["lr"],
                  "rlr": model_response_optimizer.param_groups[0]["lr"]}

    wandb.log(epoch_dict)
    return val_loss


def stn_per_epoch_evaluate(current_epoch, train_loss=None):
    def evaluate(loader):
        model.eval()
        correct = total = loss = 0.
        with torch.no_grad():
            for data in loader:
                images, labels = data[0].to(device), data[1].to(device)
                repeated_h_tensor = h_container.h_tensor.unsqueeze(0).repeat((images.shape[0], 1))
                pred = model(images, repeated_h_tensor, repeated_h_tensor)
                loss += F.cross_entropy(pred.float(), labels.long(), reduction="sum").item()
                hard_pred = torch.max(pred, 1)[1]
                total += labels.size(0)
                correct += (hard_pred == labels).sum().item()
        accuracy = correct / float(total)
        mean_loss = loss / float(total)
        return mean_loss, accuracy

    train_loader.dataset.reset_hyper_params()
    if train_loss is None:
        train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(valid_loader)
    tst_loss, tst_acc = evaluate(test_loader)

    print("=" * 80)
    print("Train Epoch: {} | Trn Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}"
          " | Test Loss: {:.3f} | Test Acc: {:.3f}".format(current_epoch, train_loss, val_loss,
                                                           val_acc, tst_loss, tst_acc))
    print("=" * 80)

    epoch_dict = {"epoch": current_epoch,
                  "train_loss": train_loss,
                  "val_loss": val_loss,
                  "val_acc": val_acc,
                  "test_loss": tst_loss,
                  "test_acc": tst_acc,
                  "lr": model_optimizer.param_groups[0]["lr"]}

    wandb.log(epoch_dict)
    return val_loss


evaluate_fnc = delta_stn_per_epoch_evaluate if info["delta_stn"] else stn_per_epoch_evaluate

stn_trainer = StnTrainer(stn_step_optimizer, train_loader=train_loader, valid_loader=valid_loader,
                         test_loader=test_loader, h_container=h_container, evaluate_fnc=evaluate_fnc,
                         device=device, lr_scheduler=lr_scheduler, warmup_epochs=info["warmup_epochs"],
                         total_epochs=info["total_epochs"], train_steps=info["train_steps"],
                         valid_steps=info["valid_steps"], log_interval=info["log_interval"],
                         patience=info["early_stop_patience"])

try:
    stn_trainer.train()
    evaluate_fnc(info["total_epochs"], None)
    sys.stdout.flush()

except KeyboardInterrupt:
    print("=" * 80)
    print("Exiting from training early ...")
    sys.stdout.flush()
