from utils.linearize_utils import *


class BaseStepOptimizer(object):
    """ A base step optimizer. """
    def step(self, inputs, labels, perturbed_h_tensor, augmented_inputs=None, tune_hyper=False):
        """ An iteration for the model.
        :param inputs: Input Tensor
        :param labels: Target Tensor
        :param perturbed_h_tensor: Tensor of size "batch_size x num_hyper"
        :param augmented_inputs: Augmented (perturbed hyperparameters) input Tensor
                                 with the same dimension as inputs
        :param tune_hyper: bool
        :return: Tensor, scalar tensor
        """
        raise NotImplementedError()


class StnStepOptimizer(BaseStepOptimizer):
    def __init__(self, model, model_optimizer, hyper_optimizer, scale_optimizer, criterion,
                 h_container, tune_scales, entropy_weight=0.):
        """ Initialize a class StnStepOptimizer.
        :param model: Module
        :param model_optimizer: Optimizer
        :param hyper_optimizer: Optimizer
        :param scale_optimizer: Optimizer
        :param criterion: Criterion
        :param h_container: HyperContainer
        :param tune_scales: bool
        :param entropy_weight: float
        """
        self.model = model

        self.model_optimizer = model_optimizer
        self.hyper_optimizer = hyper_optimizer
        self.scale_optimizer = scale_optimizer

        self.optimizers = [self.model_optimizer, self.hyper_optimizer, self.scale_optimizer]

        self.criterion = criterion
        self.h_container = h_container
        self.tune_scales = tune_scales
        self.entropy_weight = entropy_weight

    def zero_grad(self):
        for opt in self.optimizers:
            if opt is not None:
                opt.zero_grad()

    def step(self, inputs, labels, perturbed_h_tensor, augmented_inputs=None, tune_hyper=False):
        if not tune_hyper:
            self.model.train()

            self.zero_grad()
            if augmented_inputs is not None:
                inputs = augmented_inputs.detach()
            # For STNs, h_net and h_param are the same.
            pred = self.model(inputs, perturbed_h_tensor, perturbed_h_tensor)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.model_optimizer.step()
        else:
            # Turn off dropout, ... etc.
            self.model.eval()
            if self.tune_scales:
                self.zero_grad()
                # Detach h_param so that the gradient doesn't flow.
                pred = self.model(inputs, perturbed_h_tensor, perturbed_h_tensor.detach())
                loss = self.criterion(pred, labels) - self.entropy_weight * self.h_container.get_entropy()
                loss.backward()
                self.hyper_optimizer.step()
                self.scale_optimizer.step()
            else:
                self.zero_grad()
                # If not tuning the scale, do not need to perturb.
                r_current_hyper = self.h_container.h_tensor.unsqueeze(0).repeat((inputs.shape[0], 1))
                pred = self.model.forward(inputs, r_current_hyper, r_current_hyper.detach())
                loss = self.criterion(pred, labels)
                loss.backward()
                self.hyper_optimizer.step()
        return pred, loss


class DeltaStnStepOptimizer(BaseStepOptimizer):
    def __init__(self, model, model_general_optimizer, model_response_optimizer, hyper_optimizer,
                 scale_optimizer, criterion, h_container, tune_scales, entropy_weight=0., do_linearize=True):
        """ Initialize a class DeltaStnStepOptimizer.
        :param model: Module
        :param model_general_optimizer: Optimizer
        :param model_response_optimizer: Optimizer
        :param hyper_optimizer: Optimizer
        :param scale_optimizer: Optimizer
        :param criterion: Criterion
        :param h_container: HyperContainer
        :param tune_scales: bool
        :param entropy_weight: float
        :param do_linearize: bool
        """
        self.model = model

        self.model_general_optimizer = model_general_optimizer
        self.model_response_optimizer = model_response_optimizer
        self.hyper_optimizer = hyper_optimizer
        self.scale_optimizer = scale_optimizer

        self.optimizers = [self.model_general_optimizer, self.model_response_optimizer,
                           self.hyper_optimizer, self.scale_optimizer]

        self.criterion = criterion
        self.h_container = h_container
        self.tune_scales = tune_scales
        self.entropy_weight = entropy_weight
        self.do_linearize = do_linearize

    def zero_grad(self):
        for opt in self.optimizers:
            if opt is not None:
                opt.zero_grad()

    def step(self, inputs, labels, perturbed_h_tensor, augmented_inputs=None, tune_hyper=False):
        if not tune_hyper:
            self.model.train()

            self.zero_grad()
            r_current_hyper = self.h_container.h_tensor.unsqueeze(0).repeat((inputs.shape[0], 1))
            pert = r_current_hyper - r_current_hyper.detach()
            # Tuning general parameters w/o perturbation.
            pred = self.model(inputs, pert, r_current_hyper)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.model_general_optimizer.step()

            self.zero_grad()
            r_current_hyper = self.h_container.h_tensor.unsqueeze(0).repeat((inputs.shape[0], 1))
            pert = perturbed_h_tensor - r_current_hyper.detach()
            if augmented_inputs is not None:
                inputs = augmented_inputs.detach()

            if self.do_linearize:
                # Linearize the prediction:
                # Currently, PyTorch doesn't offer forward-mode autodiff:
                # Thus, it is slow to compute; this can be more efficient.
                # A current method to efficiently perform this is to use JAX.
                respond_pred, jvp_pred = jvp(self.model, inputs, r_current_hyper,
                                             perturbed_h_tensor, pert)
                respond_pred = respond_pred + jvp_pred
                # Another function is to do (will return same output):
                # respond_pred = self.model(inputs, r_current_hyper - r_current_hyper.detach(),
                #                           perturbed_h_tensor)
                # react = rop(respond_pred, r_current_hyper, pert)
                # respond_pred = respond_pred + react
            else:
                # If not doing linearization, no need to perform jvp.
                respond_pred = self.model(inputs, pert, perturbed_h_tensor)
            reaction_loss = self.criterion(respond_pred, labels)
            reaction_loss.backward()
            self.model_response_optimizer.step()

        else:
            # Turn off dropout, ... etc.
            self.model.eval()
            if self.tune_scales:
                self.zero_grad()
                r_current_hyper = self.h_container.h_tensor.unsqueeze(0).repeat((inputs.shape[0], 1))
                # Detach h_param so that the gradient doesn't flow.
                pred = self.model.forward(inputs.detach(), perturbed_h_tensor - r_current_hyper.detach(),
                                          perturbed_h_tensor.detach())
                loss = self.criterion(pred, labels) - self.entropy_weight * self.h_container.get_entropy()
                loss.backward()
                self.hyper_optimizer.step()
                self.scale_optimizer.step()
            else:
                self.zero_grad()
                # If not tuning the scale, do not need to perturb.
                r_current_hyper = self.h_container.h_tensor.unsqueeze(0).repeat((inputs.shape[0], 1))
                pred = self.model.forward(inputs.detach(), r_current_hyper - r_current_hyper.detach(),
                                          r_current_hyper.detach())
                loss = self.criterion(pred, labels)
                loss.backward()
                self.hyper_optimizer.step()
        return pred, loss
