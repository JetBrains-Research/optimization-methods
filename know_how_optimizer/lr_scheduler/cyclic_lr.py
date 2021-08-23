from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, min_lr, max_lr, cycle_len, gamma=1., last_epoch=-1, verbose=False, start_from=None, swa=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.swa = swa

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        self.cycle_len = cycle_len
        self.start_from = start_from if start_from is not None else 0
        self.min_lrs = self._format_param('min_lr', optimizer, min_lr)

        self.gamma = gamma

        super(CyclicLR, self).__init__(optimizer, last_epoch, verbose)

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

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        iteration = self.last_epoch

        t = ((iteration - self.start_from) %
             self.cycle_len + 1) / self.cycle_len

        if iteration < self.start_from:
            t = 0
        if self.swa and t == 1.:
            print("update_swa")
            self.optimizer.update_swa()

        lrs = []
        for min_lr, max_lr in zip(self.min_lrs, self.max_lrs):
            if self.swa and iteration < self.start_from:
                lr = max_lr * self.gamma**(iteration // self.cycle_len)
            else:
                lr = (1 - t) * (self.gamma**(self.start_from // self.cycle_len)
                                if iteration >= self.start_from else 1) * max_lr + t * min_lr
            lrs.append(lr)
