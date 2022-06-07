from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class MyCyclicLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 min_lr,
                 max_lr,
                 cycle_len,
                 gamma,
                 last_epoch=-1,
                 verbose=False,
                 start_from=None, swa=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.swa = swa

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)
        self.cycle_len = cycle_len
        self.start_from = 0 if start_from is None else start_from
        self.min_lrs = self._format_param('min_lr', optimizer, min_lr)
        self.gamma = gamma

        super(MyCyclicLR, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        iteration = self.last_epoch

        lrs = []
        for min_lr, max_lr in zip(self.min_lrs, self.max_lrs):
            if iteration < self.start_from:
                lr = min_lr + (max_lr - min_lr) * (float(iteration) / self.start_from)
            else:
                lr = max_lr * self.gamma**(iteration // self.cycle_len)
            lrs.append(lr)

        return lrs