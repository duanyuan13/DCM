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


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
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


class DistModule(nn.Module):
    def __init__(self, fx=3214.61823, bl=406):
        super(DistModule, self).__init__()
        self.coefficient = fx * bl / 1000

    def forward(self, offset_inputs, xx_inputs):
        """
        Args:
            offset_inputs: [B, 2]
            xx_inputs: [B, 2]
        """
        left_x = xx_inputs[:, 0] + offset_inputs[:, 0]
        right_x = xx_inputs[:, 1] + offset_inputs[:, 1]
        disparity = 1280*torch.abs(left_x-right_x)
        # print('disparity:', disparity.min().item(), disparity.max().item(), disparity.mean().item())
        return torch.div(self.coefficient, disparity)


class DistModuleDynamic(nn.Module):
    def __init__(self, fx=3214.61823, bl=406):
        super(DistModuleDynamic, self).__init__()
        self.coefficient = fx * bl / 1000

    def forward(self, offset_inputs, xx_inputs):
        """
        Args:
            offset_inputs: [B, 2]
            xx_inputs: [B, 2]
        """
        left_x = xx_inputs[:, 0] + offset_inputs[:, 0]
        right_x = xx_inputs[:, 1] + offset_inputs[:, 1]
        disparity = 1280*torch.abs(left_x-right_x)
        # print('disparity:', disparity.min().item(), disparity.max().item(), disparity.mean().item())
        return torch.div(self.coefficient, disparity), left_x, right_x









