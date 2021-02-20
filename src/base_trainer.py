from utils.data_utils import next_batch
from torch.optim.lr_scheduler import *
from utils.eval_utils import AverageMeter

import wandb

_available_lr_scheduler = [LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau]


class StnTrainer(object):
    """ A STN trainer that trains the model on a dataset."""
    def __init__(self, step_optimizer, train_loader, valid_loader, test_loader,
                 evaluate_fnc, h_container, lr_scheduler, warmup_epochs, total_epochs, device=None,
                 train_steps=5, valid_steps=1, log_interval=10, patience=None):
        """ Initialize a class StnTrainer.
        :param step_optimizer: BaseStepOptimizer
        :param train_loader: DataLoader
        :param valid_loader: DataLoader
        :param test_loader: DataLoader
        :param evaluate_fnc: function
        :param h_container: HyperContainer
        :param lr_scheduler: Scheduler
        :param warmup_epochs: int
        :param total_epochs: int
        :param device: Device
        :param train_steps: int
        :param valid_steps: int
        :param log_interval: int
        :param patience: int
        """
        self.step_optimizer = step_optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler

        # Learning rate schedules may receive a list.
        if not isinstance(self.lr_scheduler, list):
            self.lr_scheduler = [self.lr_scheduler]
        for clr in self.lr_scheduler:
            is_valid_lr_scheduler = False
            for lrs in _available_lr_scheduler:
                if isinstance(clr, lrs):
                    is_valid_lr_scheduler = True
                    break
            if not is_valid_lr_scheduler and clr is not None:
                raise Exception("Not a valid lr scheduler. "
                                "Please select {}".format(str(_available_lr_scheduler)))

        self.evaluate_fnc = evaluate_fnc
        self.h_container = h_container
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.train_steps = train_steps
        self.valid_steps = valid_steps
        self.log_interval = log_interval
        self.patience = patience
        if device is not None:
            self.device = device
        else:
            # If device is not set, just use what is being used for model.
            self.device = self.step_optimizer.model.device

    def lr_step(self, val_loss):
        for lrs in self.lr_scheduler:
            if lrs is None:
                continue
            try:
                lrs.step(val_loss, epoch=None)
            except:
                lrs.step()

    def train(self):
        """ Train the network.
        :return: None
        """
        train_iter = iter(self.train_loader)
        valid_iter = iter(self.valid_loader)

        global_step = warmup_step = 0
        train_step = valid_step = 0
        train_epoch = valid_epoch = 0
        curr_train_epoch = 0

        # Keep track of losses.
        losses = AverageMeter()
        val_losses = []
        best_val_loss = float("inf")
        best_val_epoch = 0
        try:
            self.evaluate_fnc(train_epoch)
        except:
            raise Exception("Please check your evaluation function {}.".format(str(self.evaluate_fnc)))

        while train_epoch < self.warmup_epochs:
            # Reset the data augmentation parameters.
            self.train_loader.dataset.reset_hyper_params()

            perturbed_h_tensor = self.h_container.get_perturbed_hyper(self.train_loader.batch_size)

            # Set the data augmentation hyperparameters.
            self.train_loader.dataset.set_h_container(self.h_container, perturbed_h_tensor)
            inputs, augmented_inputs, labels, train_iter, train_epoch = \
                next_batch(train_iter, self.train_loader, train_epoch, self.device)

            if curr_train_epoch != train_epoch:
                # When train_epoch changes, evaluate validation & test losses.
                val_loss = self.evaluate_fnc(train_epoch, losses.avg)
                val_losses.append(val_loss)

                losses.reset()
                curr_train_epoch = train_epoch

                self.lr_step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = curr_train_epoch
                wandb.log({
                        "best_val_loss": best_val_loss,
                        "best_val_epoch": best_val_epoch})

            # Taking care of the last batch.
            if inputs.size(0) != self.train_loader.batch_size:
                perturbed_h_tensor = perturbed_h_tensor[:inputs.size(0), :]

            _, loss = self.step_optimizer.step(inputs, labels, perturbed_h_tensor=perturbed_h_tensor,
                                               augmented_inputs=augmented_inputs, tune_hyper=False)
            losses.update(loss.item(), inputs.size(0))

            if warmup_step % self.log_interval == 0 and global_step > 0:
                print("Global Step: {} Train Epoch: {} Warmup step: {} Loss: {:.3f}".format(
                    global_step, train_epoch, warmup_step, loss))

            warmup_step += 1
            global_step += 1

        print("Warm-up finished.")
        if self.patience is None:
            self.patience = self.total_epochs

        patience_elapsed = 0
        while patience_elapsed < self.patience and train_epoch < self.total_epochs:
            for _ in range(self.train_steps):
                # Perform training steps:
                self.train_loader.dataset.reset_hyper_params()
                perturbed_h_tensor = self.h_container.get_perturbed_hyper(self.train_loader.batch_size)
                self.train_loader.dataset.set_h_container(self.h_container, perturbed_h_tensor)
                inputs, augmented_inputs, labels, train_iter, train_epoch = \
                    next_batch(train_iter, self.train_loader, train_epoch, self.device)

                if curr_train_epoch != train_epoch:
                    val_loss = self.evaluate_fnc(train_epoch, losses.avg)
                    val_losses.append(val_loss)
                    losses.reset()
                    curr_train_epoch = train_epoch

                    self.lr_step(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_epoch = curr_train_epoch
                        patience_elapsed = 0
                    else:
                        patience_elapsed += 1
                    wandb.log(
                        {"best_val_loss": best_val_loss, "best_val_epoch": best_val_epoch}
                    )

                # Again, take care of the last batch.
                if inputs.size(0) != self.train_loader.batch_size:
                    perturbed_h_tensor = perturbed_h_tensor[:inputs.size(0), :]

                _, loss = self.step_optimizer.step(inputs, labels, perturbed_h_tensor=perturbed_h_tensor,
                                                   augmented_inputs=augmented_inputs, tune_hyper=False)
                losses.update(loss.item(), inputs.size(0))

                if train_step % self.log_interval == 0 and global_step > 0:
                    print(
                        "Train - Global Step: {} Train Epoch: {} Train step:{} "
                        "Loss: {:.3f}".format(
                            global_step, train_epoch, train_step, loss))
                train_step += 1
                global_step += 1

            for _ in range(self.valid_steps):
                inputs, _, labels, valid_iter, valid_epoch = \
                    next_batch(valid_iter, self.valid_loader, valid_epoch, self.device)
                perturbed_h_tensor = self.h_container.get_perturbed_hyper(inputs.size(0))

                _, loss = self.step_optimizer.step(inputs, labels, perturbed_h_tensor=perturbed_h_tensor,
                                                   augmented_inputs=None, tune_hyper=True)

                if valid_step % self.log_interval == 0 and global_step > 0:
                    print(
                        "Valid - Global Step: {} Valid Epoch: {} Valid step:{} "
                        "Loss: {:.3f}".format(global_step, valid_epoch, valid_step, loss))

                wandb.log(self.h_container.generate_summary())
                valid_step += 1
                global_step += 1
