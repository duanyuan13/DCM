"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


def get_model(config):
    """
    Return a model object

    Args:
        config (config node object): config

    Returns:
        model (torch.nn.Module object): pytorch model
    """
    if config.name == 'DistModule':
        from models.pcm.pcm import DistModule
        model = DistModule()

    elif config.name == 'DistModuleDynamic':
        from models.pcm.pcm import DistModuleDynamic
        model = DistModuleDynamic()

    elif config.name == 'Mixer':
        from models.pcm.pcm import MLPMixer
        model = MLPMixer(image_size=(4, 1),
                         channels=1,
                         patch_size=1,
                         dim=config.dim,
                         depth=config.depth,
                         dropout=config.dropout,
                         num_classes=1)

    elif config.name == 'MixerDynamic':
        from models.pcm.pcm import MLPMixer
        model = MLPMixer(image_size=(4, 1),
                         channels=1,
                         patch_size=1,
                         dim=config.dim,
                         depth=config.depth,
                         dropout=config.dropout,
                         num_classes=2)
    else:
        logging.getLogger('Get Model').error(
            'Model for %s not implemented', config.name)
        raise NotImplementedError

    return model



