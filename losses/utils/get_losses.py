"""
Helper functions for get module or functional losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
sys.path.append('../../')


def get_loss_module(config):
    """
    Return a nn.Module object of loss

    Args:
        config (config node object)
    Returns:
        loss (torch.nn.Module): a loss module
    """
    if config.loss.loss_name == 'mse':
        from torch.nn import MSELoss
        loss = MSELoss(reduction=config.loss.reduction)

    elif config.loss.loss_name == 'cross_entropy':
        from torch.nn import CrossEntropyLoss
        loss = CrossEntropyLoss()

    elif config.loss.loss_name == 'smoothl1':
        from torch.nn import SmoothL1Loss
        loss = SmoothL1Loss(
            reduction=config.loss.reduction
        )

    elif config.loss.loss_name == 'l1loss':
        from torch.nn import L1Loss
        loss = L1Loss(
            reduction=config.loss.reduction
        )

    else:
        logging.getLogger('Get Loss Module').error(
            'Loss module for %s not implemented', config.loss.loss_name)
        raise NotImplementedError

    return loss


def get_loss_functional(config):
    """
    Return a callable object of loss

    Args:
        config (config node object): config
    Returns:
        loss (torch.nn.Module): a loss module
    """
    if config.loss.loss_name == 'mse':
        from torch.nn.functional import mse_loss
        loss_fn = mse_loss
    else:
        logging.getLogger('Get Loss Functional').error(
            'Loss module for %s not implemented', config.loss.loss_name)
        raise NotImplementedError

    return loss_fn
