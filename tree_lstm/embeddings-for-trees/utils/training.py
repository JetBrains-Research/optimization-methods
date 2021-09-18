from typing import List, Tuple, Iterable
import torch
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD, Adadelta, Adagrad, Adamax, RMSprop, LBFGS, ASGD
from torchcontrib.optim import SWA
from optimizer import SVRG, SdLBFGS, BB, RLamb, LaRAdamLamb, KFACOptimizer
import torch_optimizer as optim
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, ReduceLROnPlateau
from scheduler import MyCyclicLR
import numpy as np
from math import sqrt


def calc_grad_norm(model, batch):
    labels, graph = batch
    logits = model(graph, labels.shape[0], labels)
    model._calculate_loss(logits, labels).backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = sqrt(grad_norm)
    print('GRAD NORM:', grad_norm)
    return grad_norm


def configure_optimizers_alon(
        hyper_parameters: DictConfig, parameters: Iterable[torch.Tensor], init_grad_norm: float = 1.
) -> Tuple[List[Optimizer], List[_LRScheduler]]:
    """Create optimizers like in original Alon work
    https://github.com/tech-srl/code2seq/blob/a01076ef649d298e5f90ac2ce1f6a42f4ff49cc2/model.py#L386-L397
    :param hyper_parameters: hyper parameters
    :param parameters: model parameters for optimization
    :param init_grad_norm: norm of gradient calculated on the first train batch
    :return: list of optimizers and schedulers
    """
    assert init_grad_norm is not None
    optimizer: Optimizer
    if hyper_parameters.optimizer == "Momentum":
        # using the same momentum value as in original realization by Alon
        optimizer = SGD(
            parameters,
            hyper_parameters.learning_rate,
            momentum=0.95,
            nesterov=hyper_parameters.nesterov,
            weight_decay=hyper_parameters.weight_decay,
        )
    elif hyper_parameters.optimizer == "Nadam":
        optimizer = Nadam(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "AdaBound":
        optimizer = optim.AdaBound(parameters, lr=hyper_parameters.learning_rate,
                                   weight_decay=hyper_parameters.weight_decay,
                                   betas=(0.9, 0.999),
                                   final_lr=0.1,
                                   gamma=1e-3,
                                   eps=1e-8,
                                   amsbound=False)

    elif hyper_parameters.optimizer == "Yogi":
        optimizer = optim.Yogi(parameters, lr=hyper_parameters.learning_rate,
                               weight_decay=hyper_parameters.weight_decay,
                               betas=(0.9, 0.999),
                               eps=1e-3,
                               initial_accumulator=1e-6)

    elif hyper_parameters.optimizer == "Apollo":
        optimizer = optim.Apollo(parameters, lr=hyper_parameters.learning_rate,
                                 weight_decay=hyper_parameters.weight_decay,
                                 beta=0.9,
                                 eps=1e-4,
                                 warmup=0,
                                 init_lr=0.01
                                 )

    elif hyper_parameters.optimizer == "DiffGrad":
        optimizer = optim.DiffGrad(parameters, lr=hyper_parameters.learning_rate,
                                   weight_decay=hyper_parameters.weight_decay,
                                   betas=(0.9, 0.999),
                                   eps=1e-8
                                   )

    elif hyper_parameters.optimizer == "NovoGrad":
        optimizer = optim.NovoGrad(parameters, lr=hyper_parameters.learning_rate,
                                   weight_decay=hyper_parameters.weight_decay,
                                   betas=(0.9, 0.999),
                                   eps=1e-8,
                                   grad_averaging=False,
                                   amsgrad=False
                                   )

    elif hyper_parameters.optimizer == "SWA":
        base_opt = SGD(
            parameters,
            hyper_parameters.learning_rate,
            momentum=0.95,
            nesterov=hyper_parameters.nesterov,
            weight_decay=hyper_parameters.weight_decay,
        )
        optimizer = SWA(base_opt)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "ASGD":
        optimizer = ASGD(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Adam":
        optimizer = Adam(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Lookahead":
        sgd = SGD(
            parameters,
            hyper_parameters.learning_rate,
            momentum=0.95,
            nesterov=hyper_parameters.nesterov,
            weight_decay=hyper_parameters.weight_decay,
        )
        optimizer = optim.Lookahead(sgd, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lookahead_Adam":
        adam = Adam(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay,
                            betas=(0.9, 0.999),
                            eps=1e-8
                            )
        optimizer = optim.Lookahead(adam, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lookahead_RAdam":
        radam = optim.RAdam(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay,
                            betas=(0.9, 0.999),
                            eps=1e-8
                            )
        optimizer = optim.Lookahead(radam, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lookahead_Lamb":
        lamb = optim.Lamb(
            parameters, lr=hyper_parameters.learning_rate * init_grad_norm,
            weight_decay=hyper_parameters.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        optimizer = optim.Lookahead(lamb, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lookahead_Yogi":
        yogi = optim.Yogi(parameters, lr=hyper_parameters.learning_rate,
                               weight_decay=hyper_parameters.weight_decay,
                               betas=(0.9, 0.999),
                               eps=1e-3,
                               initial_accumulator=1e-6)
        optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lookahead_DiffGrad":
        diffgrad = optim.DiffGrad(parameters, lr=hyper_parameters.learning_rate,
                                   weight_decay=hyper_parameters.weight_decay,
                                   betas=(0.9, 0.999),
                                   eps=1e-8
                                   )
        optimizer = optim.Lookahead(diffgrad, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lookahead_Adamax":
        adamax = Adamax(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
        optimizer = optim.Lookahead(adamax, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Adadelta":
        optimizer = Adadelta(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "A2GradExp":
        optimizer = optim.A2GradExp(parameters, lr=hyper_parameters.learning_rate,
                                    beta=10.0,
                                    lips=10.0,
                                    rho=0.5
                                    )

    elif hyper_parameters.optimizer == "Adagrad":
        optimizer = Adagrad(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Adamax":
        optimizer = Adamax(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "RMSprop":
        optimizer = RMSprop(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "RAdam":
        optimizer = optim.RAdam(
            parameters, lr=hyper_parameters.learning_rate,
            weight_decay=hyper_parameters.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    elif hyper_parameters.optimizer == "Lamb":
        optimizer = optim.Lamb(
            parameters, lr=hyper_parameters.learning_rate * init_grad_norm,
            weight_decay=hyper_parameters.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    elif hyper_parameters.optimizer == "RLamb":
        optimizer = RLamb(
            parameters, lr=hyper_parameters.learning_rate,
            weight_decay=hyper_parameters.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    elif hyper_parameters.optimizer == "LBFGS":
        optimizer = LBFGS(parameters, hyper_parameters.learning_rate)

    elif hyper_parameters.optimizer == "SdLBFGS":
        optimizer = SdLBFGS(parameters, lr=hyper_parameters.learning_rate, lr_decay=hyper_parameters.lr_decay,
                            weight_decay=hyper_parameters.weight_decay, max_iter=hyper_parameters.max_iter,
                            history_size=hyper_parameters.history_size)

    elif hyper_parameters.optimizer == "SVRG":
        optimizer = SVRG(parameters, hyper_parameters.learning_rate, freq=80)

    elif hyper_parameters.optimizer == "BB":
        optimizer = BB(parameters, lr=hyper_parameters.learning_rate, steps=50, beta=0.01, max_lr=0.1, min_lr=0.001)

    else:
        raise ValueError(f"Unknown optimizer name: {hyper_parameters.optimizer}")

    if hyper_parameters.strategy == "decay":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: hyper_parameters.lr_decay_gamma ** epoch)
        # scheduler = {
        #     'scheduler': LambdaLR(optimizer, lr_lambda=lambda epoch: hyper_parameters.lr_decay_gamma ** epoch),
        #     'interval': 'epoch',  # or 'epoch'
        #     'frequency': 1
        # }
    elif hyper_parameters.strategy == 'warmup':
        scheduler = {
            'scheduler': LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(
                    hyper_parameters.lr_decay_gamma ** (step // hyper_parameters.steps_in_epoch),
                    (step + 1) / hyper_parameters.warmup_steps
                )
            ),
            'interval': 'epoch',  # or 'epoch'
            'frequency': 1
        }
    elif hyper_parameters.strategy == "reduce_on_plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.95, patience=50)
        scheduler = {
           'scheduler': lr_scheduler,
           'reduce_on_plateau': True,
           'interval': 'step',
           'monitor': 'train/loss',
        }
    elif hyper_parameters.strategy == "cyclic":
        scheduler = {
            'scheduler': MyCyclicLR(optimizer, min_lr=hyper_parameters.min_lr, max_lr=hyper_parameters.max_lr,
                                    cycle_len=hyper_parameters.cycle_len, gamma=hyper_parameters.lr_decay_gamma,
                                    start_from=hyper_parameters.start_from, swa=(hyper_parameters.optimizer == "SWA")),
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
    else:
        raise ValueError('No such learning rate strategy')
    return [optimizer], [scheduler]


def segment_sizes_to_slices(sizes: torch.Tensor) -> List:
    cum_sums = torch.cumsum(sizes, dim=0)
    slices = [slice(0, cum_sums[0])]
    slices += [slice(start, end) for start, end in zip(cum_sums[:-1], cum_sums[1:])]
    return slices


def cut_encoded_data(
    encoded_data: torch.Tensor, samples_per_label: torch.LongTensor, mask_value: float = -1e9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut encoded data into batches
    :param encoded_data: [n contexts; units]
    :param samples_per_label: [batch size]
    :param mask_value:
    :return: [batch size; max context len; units], [batch size; max context len]
    """
    batch_size = len(samples_per_label)
    max_context_len = max(samples_per_label)
    batched_contexts = encoded_data.new_zeros((batch_size, max_context_len, encoded_data.shape[-1]))
    attention_mask = encoded_data.new_zeros((batch_size, max_context_len))
    context_slices = segment_sizes_to_slices(samples_per_label)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, samples_per_label)):
        batched_contexts[i, :cur_size] = encoded_data[cur_slice]
        attention_mask[i, cur_size:] = mask_value
    return batched_contexts, attention_mask


def init_weights_normal(m):
    """Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution."""
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname == 'Linear':
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname == 'LSTM':
        num_layers = m.num_layers
        sigma = 1 / np.sqrt(m.hidden_size)
        for i in range(num_layers):
            getattr(m, 'weight_ih_l' + str(i)).data.normal_(0.0, sigma)
            getattr(m, 'weight_hh_l' + str(i)).data.normal_(0.0, sigma)
            if m.bias:
                getattr(m, 'bias_ih_l' + str(i)).data.fill_(0)
                getattr(m, 'bias_hh_l' + str(i)).data.fill_(0)


def init_weights_const(m, value=0):
    if classname == 'Linear':
        m.weight.data.fill_(value)
        if m.bias is not None:
            m.bias.data.fill_(value)
    elif classname == 'LSTM':
        num_layers = m.num_layers
        for i in range(num_layers):
            getattr(m, 'weight_ih_l' + str(i)).data.fill_(value)
            getattr(m, 'weight_hh_l' + str(i)).data.fill_(value)
            if m.bias:
                getattr(m, 'bias_ih_l' + str(i)).data.fill_(value)
                getattr(m, 'bias_hh_l' + str(i)).data.fill_(value)
    elif classname == 'Embedding':
        m.weight.data.fill_(value)
