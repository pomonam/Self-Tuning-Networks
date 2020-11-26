# Self-Tuning Networks (STNs)
This repository contains a clean-up code for Self-Tuning Networks (STNs) and Delta Self-Tuning Networks (![d]-STNs). 
The original repository for Self-Tuning Networks can be found [here](https://github.com/asteroidhouse/self-tuning-networks).

Papers: 
- [Delta-STN: Efficient Bilevel Optimization for Neural Networks using Structured Response Jacobian](https://arxiv.org/abs/2010.13514)
- [Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Structured Best-Response Functions](https://arxiv.org/abs/1903.03088)

## Requirements
The code was implemented & tested in Python 3.6. All required modules are listed in `requirements.txt` and can be installed with the following command:
```
pip install -r requirements.txt
```
In addition, please install [PyTorch](https://pytorch.org/) version 1.5.1 (or >= 1.5.0). We plan to release the JAX version of the code as well.

## How to use STNs for Custom Projects?
Self-Tuning Networks can be easily applied to any architectures, datasets, and regularization hyperparameters. Please follow these steps to use STNs for your custom projects.

1. Define your model inheriting from `StnModel` using layers from `\layers`. Specify how your models interact with hyperparameters. 
```
class StnTwoLayerMLP(StnModel):
    # Inherit from StnModel.
    def __init__(self, input_dim, output_dim, num_hyper, h_container, use_bias=True):
        super(StnTwoLayerMLP, self).__init__()
        self.input_dim = input_dim
        self.layer_structure = [input_dim, 1200, 1200, output_dim]
        self.num_hyper = num_hyper
        # h_container (HyperContainer) contains all information about hyperparameters.
        self.h_container = h_container
        self.use_bias = use_bias

        # Use StnLinear instead of nn.Linear.
        self.layers = nn.ModuleList(
            [StnLinear(self.layer_structure[i], self.layer_structure[i + 1],
                       num_hyper=num_hyper, bias=use_bias)
             for i in range(len(self.layer_structure) - 1)]
        )
    
    # This method must be defined; it should return a list containing all layers.
    def get_layers(self):
        return self.layers

    def forward(self, x, h_net, h_tensor):
        # Forward method requires h_net and h_tensor.
        # For STNs, h_net and h_tensor are the same. 
        # However, for Delta-STNs, they differ as h_net requires centering.
        x = x.view(-1, self.input_dim)
        
        # Apply dropout for each batch using parameters from h_tensor (perturbed dropout).
        if "dropout0" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout0"), self.training)

        # STN layers requires one additional input h_net.
        x = self.layers[0](x, h_net)
        x = F.relu(x)
        if "dropout1" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout1"), self.training)

        x = self.layers[1](x, h_net)
        x = F.relu(x)
        if "dropout2" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout2"), self.training)

        x = self.layers[2](x, h_net)
        return x
```
2. To tune data augmentation parameters, define your dataset class. See more examples in `/data`. If you wish not to tune data augmentation parameters, you can define a class with these methods:
```
class StnMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(StnMNIST, self).__init__(*args, **kwargs)

    def set_h_container(self, h_container, perturbed_h_tensor):
        pass

    def reset_hyper_params(self):
        pass
```
3. In your training script, initialize a class `HyperContainer` from `/hyper` and register all hyperparameters using a method `.register`.
```
h_container = HyperContainer(device)

h_container.register("dropout1",
                     info["initial_dropout_value"],
                     info["initial_dropout_scale"],
                     min_range=0., max_range=0.95,
                     discrete=False, same_perturb_mb=False)

h_container.register("dropout2",
                     info["initial_dropout_value"],
                     info["initial_dropout_scale"],
                     min_range=0., max_range=0.95,
                     discrete=False, same_perturb_mb=False)
```

4. Choose your desired optimizers and initialize `StnStepOptimizer` (for STNs) or `DeltaStnStepOptimizer` (for Delta-STNs).
```
model_optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr, momentum=0.9)
hyper_optimizer = torch.optim.RMSprop([h_container.h_tensor], lr=args.valid_lr)
scale_optimizer = torch.optim.RMSprop([h_container.h_scale], lr=args.scale_lr)

stn_step_optimizer = StnStepOptimizer(model, model_optimizer, hyper_optimizer, scale_optimizer, criterion,
                                      h_container, info["tune_scales"], info["entropy_weight"])
```
5. Initialize a trainer with your desired configurations and train the model.
```
stn_trainer = StnTrainer(stn_step_optimizer, train_loader=train_loader, valid_loader=valid_loader,
                         test_loader=test_loader, h_container=h_container, evaluate_fnc=evaluate_fnc,
                         device=device, lr_scheduler=None, logger=logger, warmup_epochs=info["warmup_epochs"],
                         total_epochs=info["total_epochs"], train_steps=5, valid_steps=1,
                         log_interval=10, patience=None)
stn_trainer.train()
```

## Examples
The repository contains examples to reproduce results from the Delta-STN paper.

1. Multilayer Perceptron experiment on MNIST:
- STN
```
python examples/mlp/train.py --entropy_weight 1e-3 --tune_scales --experiment_name mlp_ts_ew1e-3
```
- Delta-STN
```
python examples/mlp/train.py --delta_stn --linearize --entropy_weight 1e-3 --tune_scales --experiment_name mlp_ts_lin_ew1e-3
```
2. Simple CNN experiment on FashionMNIST: 
- STN
```
python examples/simple_cnn/train.py --entropy_weight 1e-3 --tune_scales --experiment_name cnn_ts_ew1e-3
```
- Delta-STN
```
python examples/simple_cnn/train.py --delta_stn --linearize --entropy_weight 1e-3 --tune_scales --experiment_name cnn_ts_lin_ew1e-3
```
3. VGG16 experiment on CIFAR10: 
- STN
```
python examples/vgg/train.py --entropy_weight 1e-3 --tune_scales --experiment_name vgg_ts_ew1e-3
```
- Delta-STN
```
python examples/vgg/train.py --delta_stn --linearize --entropy_weight 1e-4 --tune_scales --experiment_name vgg_ts_lin_ew1e-4
```

## Visualization
The repository supports [wandb](https://www.wandb.com/) visualization. You can either visualize your training online using wandb or use TensorBoard visualization with the following command:
```
tensorboard --logdir=examples/mlp/runs/
```

## Citation
To cite this work, please use:
```
@article{bae2020delta,
  title={Delta-STN: Efficient Bilevel Optimization for Neural Networks using Structured Response Jacobians},
  author={Bae, Juhan and Grosse, Roger B},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
@article{mackay2019self,
  title={Self-tuning networks: Bilevel optimization of hyperparameters using structured best-response functions},
  author={MacKay, Matthew and Vicol, Paul and Lorraine, Jon and Duvenaud, David and Grosse, Roger},
  journal={arXiv preprint arXiv:1903.03088},
  year={2019}
}
```

## Contributors
- [Juhan Bae](https://github.com/pomonam)

If you have any questions or suggestions, please feel free to contact me via jbae at cs dot toronto dot edu.

[d]: http://chart.apis.google.com/chart?cht=tx&chl=\Delta
