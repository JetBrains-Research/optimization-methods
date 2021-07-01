import math

import torch
from torch.optim.optimizer import Optimizer
from torch_optimizer.types import Betas2, OptFloat, OptLossClosure, Params


class RLamb(Optimizer):
    """
    Implements RAdam/Lamb cross method

    Arguments:
    params: iterable of parameters to optimize or dicts defining
        parameter groups
    lr: learning rate (default: 1e-3)
    betas: coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps: term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay: weight decay (L2 penalty) (default: 0)
    clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
        set to a high value to avoid it (e.g 10e3)
    radam: always use trust ratio = 1, which turns this
        into RAdam. Useful for comparison purposes. (default: False)
    """

    def __init__(
            self,
            params: Params,
            lr: float = 1e-3,
            betas: Betas2 = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            clamp_value: float = 10,
            radam: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                        param['betas'][0] != betas[0]
                        or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)],
                    )
        self.clamp_value = clamp_value
        self.radam = radam

        super(RLamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RLamb, self).__setstate__(state)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    msg = (
                        'RLamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (
                            1 - beta2_t
                    )
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                                lr
                                * math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                                / (1 - beta1 ** state['step'])
                        )
                    else:
                        step_size = lr / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if weight_decay != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-weight_decay * lr)

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                if N_sma >= 5:
                    radam_step = exp_avg / exp_avg_sq.sqrt().add(eps)
                else:
                    radam_step = exp_avg

                radam_norm = torch.norm(radam_step)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm
                state['weight_norm'] = weight_norm
                state['radam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio
                if self.radam:
                    trust_ratio = 1

                p_data_fp32.add_(radam_step, alpha=-step_size * trust_ratio)
                p.data.copy_(p_data_fp32)

        return loss
