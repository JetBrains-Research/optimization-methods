from typing import List, Tuple, Iterable
from omegaconf import DictConfig
import torch
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD, Adadelta, Adagrad, Adamax, RMSprop, LBFGS, ASGD
import torch_optimizer as optim
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


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

    elif hyper_parameters.optimizer == "ASGD":
        optimizer = ASGD(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Adam":
        optimizer = Adam(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Adadelta":
        optimizer = Adadelta(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Adagrad":
        optimizer = Adagrad(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "Adamax":
        optimizer = Adamax(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "RMSprop":
        optimizer = RMSprop(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)

    elif hyper_parameters.optimizer == "RAdam":
        optimizer = optim.RAdam(parameters, lr=hyper_parameters.learning_rate,
                                weight_decay=hyper_parameters.weight_decay,
                                betas=(0.9, 0.999),
                                eps=1e-8
                                )

    elif hyper_parameters.optimizer == "LBFGS":
        optimizer = LBFGS(parameters, hyper_parameters.learning_rate)

    else:
        raise ValueError(f"Unknown optimizer name: {hyper_parameters.optimizer}")
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: hyper_parameters.decay_gamma ** epoch)
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
