from typing import Iterable

import torch
from omegaconf import DictConfig
from torch.optim import SGD, Adam
from torch_optimizer import Lamb, RAdam, Lookahead

# from know_how_optimizer import BB
# from know_how_optimizer.lr_scheduler import CyclicLR

from torch.optim.lr_scheduler import LambdaLR


def configure_optimizers_custom(hyper_parameters: DictConfig, parameters: Iterable[torch.Tensor]):
    optimizer = SGD(parameters, hyper_parameters.learning_rate,
                    weight_decay=hyper_parameters.weight_decay)

    if hyper_parameters.optimizer == "SGD":
        optimizer = SGD(parameters, hyper_parameters.learning_rate,
                        weight_decay=hyper_parameters.weight_decay)

    if hyper_parameters.optimizer == "LaSGD":
        sgd = SGD(parameters, hyper_parameters.learning_rate,
                  weight_decay=hyper_parameters.weight_decay)
        optimizer = Lookahead(sgd, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Adam":
        optimizer = Adam(parameters, hyper_parameters.learning_rate,
                         weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "LaAdam":
        adam = Adam(parameters, hyper_parameters.learning_rate,
                    weight_decay=hyper_parameters.weight_decay)
        optimizer = Lookahead(adam, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "Lamb":
        optimizer = Lamb(parameters, lr=hyper_parameters.learning_rate,
                         weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "LaLamb":
        lamb = Lamb(parameters, lr=hyper_parameters.learning_rate,
                    weight_decay=hyper_parameters.weight_decay)
        optimizer = Lookahead(lamb, k=5, alpha=0.5)
        optimizer.defaults = []

    elif hyper_parameters.optimizer == "RAdam":
        optimizer = RAdam(parameters, lr=hyper_parameters.learning_rate,
                          weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "LaRAdam":
        radam = RAdam(parameters, lr=hyper_parameters.learning_rate,
                      weight_decay=hyper_parameters.weight_decay)
        optimizer = Lookahead(radam, k=5, alpha=0.5)
        optimizer.defaults = []

    # elif hyper_parameters.optimizer == "BB":
    #     optimizer = BB(parameters, lr=hyper_parameters.learning_rate,
    #                    steps=50, beta=0.01, max_lr=0.1, min_lr=0.001)

    if hyper_parameters.strategy == "decay":
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=lambda epoch: hyper_parameters.decay_gamma ** epoch),
            'interval': 'epoch',
            'frequency': 1
        }
    # elif hyper_parameters.strategy == "cyclic":
    #     scheduler = {
    #         'scheduler': CyclicLR(optimizer, min_lr=hyper_parameters.min_lr, max_lr=hyper_parameters.max_lr,
    #                               cycle_len=hyper_parameters.cycle_len, start_from=hyper_parameters.start_from,
    #                               swa=True),
    #         'interval': 'step',
    #         'frequency': 1
    #     }
    else:
        raise ValueError('No such learning rate strategy')

    return [optimizer], [scheduler]
