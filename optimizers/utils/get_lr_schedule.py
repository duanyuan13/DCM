"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

from torch.optim.lr_scheduler import LambdaLR


def get_lr_schedule(config, optimizer):
    """
    Return a learning rate schedule object

    Args:
        config (config node object): config

    Returns:
        optimizer (torch.optim.Optimizer object): pytorch optimizer
    """

    if config.optimizer.lr_scheduler.lr_scheduler_name == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=config.optimizer.lr_scheduler.factor,
            patience=config.optimizer.lr_scheduler.patience,
            min_lr=config.optimizer.lr_scheduler.min_lr,
            verbose=True, threshold=0.0001, threshold_mode='rel',
            cooldown=0, eps=1e-08,
        )

    elif config.optimizer.lr_scheduler.lr_scheduler_name == 'cyclic':
        from torch.optim.lr_scheduler import CyclicLR
        scheduler = CyclicLR(
            optimizer=optimizer,
            base_lr=config.optimizer.lr_scheduler.base_lr,
            max_lr=config.optimizer.lr_scheduler.max_lr,
            step_size_up=2000,
            step_size_down=None,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle',
            cycle_momentum=False,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1
        )

    elif config.optimizer.lr_scheduler.lr_scheduler_name == 'cos':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.optimizer.max_epoch,
            eta_min=config.optimizer.lr_scheduler.min_lr,
        )

    elif config.optimizer.lr_scheduler.lr_scheduler_name == 'warmup_cos':
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=config.optimizer.lr_scheduler.warmup_steps,
            t_total=config.optimizer.max_epoch)

    elif config.optimizer.lr_scheduler.lr_scheduler_name == 'none':
        return None

    else:
        logging.getLogger('Get LR Schedule').error(
            'Schedule for %s not implemented',
            config.lr_scheduler.lr_scheduler_name)
        raise NotImplementedError

    return scheduler


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        # print('cos:', max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))