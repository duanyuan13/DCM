import torch


def compute_abs_rel_error(distance, gt_distance):
    """
    compute absolute relative error.
    the formula is: abs_rel_error = abs(distance - gt_distance) / gt_distance
    """
    return torch.abs(distance - gt_distance) / gt_distance


def compute_square_rel_error(distance, gt_distance):
    """
    compute square relative error.
    the formula is: sq_rel_error = abs(distance - gt_distance) / gt_distance
    """
    return torch.square(distance - gt_distance) / gt_distance