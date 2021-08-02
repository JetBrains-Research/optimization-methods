import math
from collections import defaultdict
from itertools import tee

import torch
from torch.optim.optimizer import Optimizer
from torch_optimizer import RAdam, Lamb, Lookahead
from torch_optimizer.types import Betas2, OptFloat, OptLossClosure, Params


class LaRAdamLamb(Optimizer):
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
    lookahead: whether to use Lookahead envelope with RAdam in the beginning. (default: False)
    """

    def __init__(
            self,
            params: Params,
            lr: float = 1e-3,
            betas: Betas2 = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            switch_iter: int = 5000,
            lookahead: bool = True,
    ) -> None:
        params1, params2 = tee(params)
        radam = RAdam(params1, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        self.lamb = Lamb(
            params2, lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )

        self.lookahead = lookahead
        self.optimizer = Lookahead(radam, k=5, alpha=0.5) if lookahead else radam
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.switch_iter = switch_iter
        self.steps = 0
        if lookahead:
          self.defaults = []

    def __setstate__(self, state):
        super(LaRAdamLamb, self).__setstate__(state)

    def _switch_optimizers(self):
        starter_state_dict = self.optimizer.state_dict()
        corrected_state_dict = dict()
        corrected_state_dict['param_groups'] = starter_state_dict['param_groups']
        if self.lookahead:
            corrected_state_dict['state'] = starter_state_dict['slow_state']
        else:
            corrected_state_dict['state'] = starter_state_dict['state']
        self.optimizer = self.lamb
        self.optimizer.load_state_dict(state_dict=corrected_state_dict)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if self.steps == self.switch_iter:
            self._switch_optimizers()
        loss = self.optimizer.step(closure=closure)
        self.steps += 1

        return loss

    def state_dict(self):
      state_dict = super(LaRAdamLamb, self).state_dict()
      return {
          'optimizer_states': state_dict,
          'steps': self.steps
      }

    def load_state_dict(self, state_dict):
        step = state_dict['steps']
        super(Lookahead, self).load_state_dict(state_dict['optimizer_states'])
        if step >= self.switch_iter:
            self.optimizer = self.lamb
        self.optimizer.load_state_dict(state_dict['optimizer_states'])


    def zero_grad(self):
        self.optimizer.zero_grad()
