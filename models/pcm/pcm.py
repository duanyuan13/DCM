# Refs: the part of Mlp-Mixer is coming from: https://github.com/lucidrains/mlp-mixer-pytorch
import torch.nn as nn
import torch
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
        # return self.fn(x) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        # nn.Tanh(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes,
             expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


def MLPActMixer(*, image_size, channels, patch_size, dim, depth, num_classes,
                act='tanh', expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    if act == 'tanh':
        head_act = nn.Tanh()
    elif act == 'sigmoid':
        head_act = nn.Sigmoid()
    else:
        # can be added here.
        raise NotImplementedError(f'{act} is not implemented')

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes),
        head_act
    )


class DistModule(nn.Module):
    def __init__(self, fx=3.215, bl=406, img_width=1280):
        """
        Args:
            fx: focal length(in mm), put your focal length here. The focal length is calculated by fx=lenses_of_you_focal_lengh / img_sensor_size
            bl: baseline, the distance of two camera(in mm)
        """
        super(DistModule, self).__init__()
        self.coefficient = fx * bl
        self.img_width = img_width

    def forward(self, offset_inputs, xx_inputs):
        """
        Args:
            offset_inputs: [B, 2]
            xx_inputs: [B, 2]
        """
        left_x = xx_inputs[:, 0] + offset_inputs[:, 0]
        right_x = xx_inputs[:, 1] + offset_inputs[:, 1]
        disparity = self.img_width * torch.abs(left_x - right_x)
        return torch.div(self.coefficient, disparity), left_x, right_x