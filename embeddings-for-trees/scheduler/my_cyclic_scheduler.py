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

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.swa = swa

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)
        # if last_epoch == -1:
        #     for lr, group in zip(max_lrs, optimizer.param_groups):
        #         group['lr'] = lr

        self.cycle_len = cycle_len
        self.start_from = 0 if start_from is None else start_from
        self.min_lrs = self._format_param('min_lr', optimizer, min_lr)

        # step_size_up = float(step_size_up)
        # step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        # self.total_size = step_size_up + step_size_down
        # self.step_ratio = step_size_up / self.total_size

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

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        iteration = self.last_epoch

        t = ((iteration - self.start_from) % self.cycle_len + 1) / self.cycle_len

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
                lr = (1 - t) * (self.gamma**(self.start_from // self.cycle_len) if iteration >= self.start_from else 1) * max_lr + t * min_lr
            lrs.append(lr)

        return lrs
