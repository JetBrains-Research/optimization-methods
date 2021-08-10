from typing import List, Tuple, Iterable
import torch
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD, Adadelta, Adagrad, Adamax, RMSprop, LBFGS, ASGD
from torchcontrib.optim import SWA
from optimizer import SVRG, SdLBFGS, BB, RLamb
import torch_optimizer as optim
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from scheduler import MyCyclicLR


def configure_optimizers_alon(
        hyper_parameters: DictConfig, parameters: Iterable[torch.Tensor]
) -> Tuple[List[Optimizer], List[_LRScheduler]]:
    """Create optimizers like in original Alon work
    https://github.com/tech-srl/code2seq/blob/a01076ef649d298e5f90ac2ce1f6a42f4ff49cc2/model.py#L386-L397
    :param hyper_parameters: hyper parameters
    :param parameters: model parameters for optimization
    :return: list of optimizers and schedulers
    """
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
            parameters, lr=hyper_parameters.learning_rate,
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
