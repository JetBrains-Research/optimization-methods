from typing import List, Tuple, Iterable

import numpy
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer, SGD, Adam
from torch_optimizer import Lamb, RAdam, Lookahead

from know_how_optimizer import BB
from know_how_optimizer.lr_sheduler import CyclicLR

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def configure_optimizers(hyper_parameters: DictConfig, parameters: Iterable[torch.Tensor]) -> Tuple[List[Optimizer], List[_LRScheduler]]:
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

    elif hyper_parameters.optimizer == "BB":
        optimizer = BB(parameters, lr=hyper_parameters.learning_rate,
                       steps=50, beta=0.01, max_lr=0.1, min_lr=0.001)

    if hyper_parameters.strategy == "decay":
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=lambda epoch: hyper_parameters.decay_gamma ** epoch),
            'interval': 'epoch',
            'frequency': 1
        }
    elif hyper_parameters.strategy == "cyclic":
        scheduler = {
            'scheduler': CyclicLR(optimizer, min_lr=hyper_parameters.min_lr, max_lr=hyper_parameters.max_lr,
                                  cycle_len=hyper_parameters.cycle_len, start_from=hyper_parameters.start_from, swa=True), Ï
            'interval': 'step',
            'frequency': 1
        }
    else:
        raise ValueError('No such learning rate strategy')

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

    batched_contexts = encoded_contexts.new_zeros(
        (batch_size, max_context_len, encoded_contexts.shape[-1]))
    attention_mask = encoded_contexts.new_zeros((batch_size, max_context_len))

    context_slices = segment_sizes_to_slices(contexts_per_label)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, contexts_per_label)):
        batched_contexts[i, :cur_size] = encoded_contexts[cur_slice]
        attention_mask[i, cur_size:] = mask_value

    return batched_contexts, attention_mask
