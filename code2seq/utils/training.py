from typing import List, Tuple, Iterable

import numpy
import torch
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD
from torch_optimizer import Lamb, RAdam, Lookahead
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def configure_optimizers_alon(
    hyper_parameters: DictConfig, parameters: Iterable[torch.Tensor], fast_lr_lamb: bool = True
) -> Tuple[List[Optimizer], List[_LRScheduler]]:
    """Create optimizers like in original Alon work
    https://github.com/tech-srl/code2seq/blob/a01076ef649d298e5f90ac2ce1f6a42f4ff49cc2/model.py#L386-L397
    :param hyper_parameters: hyper parameters
    :param parameters: model parameters for optimization
    :return: list of optimizers and schedulers
    """
    optimizer: Optimizer
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
        if fast_lr_lamb:
            grad_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            print('grad_norm', grad_norm)
        else:
            grad_norm = 1.0
        
        optimizer = Lamb(parameters, lr=grad_norm * hyper_parameters.learning_rate,
                         weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "LaLamb":
        if fast_lr_lamb:
            grad_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            print('grad_norm', grad_norm)
        else:
            grad_norm = 1.0
        
        lamb = Lamb(parameters, lr=grad_norm * hyper_parameters.learning_rate,
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
    else:
        raise ValueError(f"Unknown optimizer name: {hyper_parameters.optimizer}")
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: hyper_parameters.decay_gamma ** epoch)
    return [optimizer], [scheduler]


def segment_sizes_to_slices(sizes: List) -> List:
    cum_sums = numpy.cumsum(sizes)
    start_of_segments = numpy.append([0], cum_sums[:-1])
    return [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]


def cut_encoded_contexts(
    encoded_contexts: torch.Tensor, contexts_per_label: List[int], mask_value: float = -1e9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut encoded contexts into batches

    :param encoded_contexts: [n contexts; units]
    :param contexts_per_label: [batch size]
    :param mask_value:
    :return: [batch size; max context len; units], [batch size; max context len]
    """
    batch_size = len(contexts_per_label)
    max_context_len = max(contexts_per_label)

    batched_contexts = encoded_contexts.new_zeros((batch_size, max_context_len, encoded_contexts.shape[-1]))
    attention_mask = encoded_contexts.new_zeros((batch_size, max_context_len))

    context_slices = segment_sizes_to_slices(contexts_per_label)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, contexts_per_label)):
        batched_contexts[i, :cur_size] = encoded_contexts[cur_slice]
        attention_mask[i, cur_size:] = mask_value

    return batched_contexts, attention_mask
