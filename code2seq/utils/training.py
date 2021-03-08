from typing import List, Tuple, Iterable

import numpy
import torch
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD, Adadelta, Adagrad, Adamax, RMSprop, LBFGS
import torch_optimizer as optim
from optimizer import Nadam, SVRG
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
    elif hyper_parameters.optimizer == "Adam":
        optimizer = Adam(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
    elif hyper_parameters.optimizer == "Nadam":
        optimizer = Nadam(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
        
    elif hyper_parameters.optimizer == "Adadelta":
        optimizer = Adadelta(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
    
    elif hyper_parameters.optimizer == "Adagrad":
        optimizer = Adagrad(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
    
    elif hyper_parameters.optimizer == "Adamax":
        optimizer = Adamax(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
    
    elif hyper_parameters.optimizer == "RMSprop":
        optimizer = RMSprop(parameters, hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay)
    
    elif hyper_parameters.optimizer == "AdaBound":
        optimizer = optim.AdaBound(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay, 
                                   betas= (0.9, 0.999),
                                   final_lr = 0.1,
                                   gamma=1e-3,
                                   eps= 1e-8,
                                   amsbound=False)
        
    elif hyper_parameters.optimizer == "Yogi":
        optimizer = optim.Yogi(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay, 
                                   betas=(0.9, 0.999),
                                    eps=1e-3,
                                    initial_accumulator=1e-6)
        
    elif hyper_parameters.optimizer == "Apollo":
        optimizer = optim.Apollo(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay, 
            beta=0.9,
            eps=1e-4,
            warmup=0,
            init_lr=0.01
        )
        
    elif hyper_parameters.optimizer == "A2GradExp":
        optimizer = optim.A2GradExp(parameters, lr=hyper_parameters.learning_rate,
    beta=10.0,
    lips=10.0,
    rho=0.5
)
        
    elif hyper_parameters.optimizer == "DiffGrad":
        optimizer = optim.DiffGrad(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay, 
    betas=(0.9, 0.999),
    eps=1e-8
)
        
    elif hyper_parameters.optimizer == "Lamb":
        optimizer = optim.Lamb(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
)
        
    elif hyper_parameters.optimizer == "RAdam":
        optimizer = optim.RAdam(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
)
        
    elif hyper_parameters.optimizer == "NovoGrad":
        optimizer = optim.NovoGrad(parameters, lr=hyper_parameters.learning_rate, weight_decay=hyper_parameters.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
    grad_averaging=False,
    amsgrad=False
)
    
    elif hyper_parameters.optimizer == "SVRG":
        optimizer = SVRG(parameters, hyper_parameters.learning_rate, freq=80)
    
    elif hyper_parameters.optimizer == "LBFGS":
        optimizer = LBFGS(parameters, hyper_parameters.learning_rate)
    
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
