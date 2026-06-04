import os
from functools import reduce
from typing import List

import torch
from torch import nn

from easytorch.utils.registry import scan_modules

from .builder import LOSS_REGISTRY
from .other_losses import LogL1Loss, LogL2Loss, CharbonnierLoss, EdgeLoss, StyleLoss, PerceptualLoss

__all__ = ['build_loss', 'LOSS_REGISTRY']


def calculate_loss(loss: nn.Module, y_pred: torch.Tensor, y_gt: torch.Tensor, **kwargs) -> torch.Tensor:
    """Uniform interface for calculating loss.

    Args:
        loss (nn.Module): Loss object.
        y_pred (torch.Tensor): Predicted image.
        y_gt (torch.Tensor): GT image.
        kwargs: other inputs.

    Return:
        loss value (torch.Tensor)
    """

    args_count = loss.forward.__code__.co_argcount
    args_name = loss.forward.__code__.co_varnames[:args_count]

    loss_kwargs = dict((k, v) for k, v in kwargs.items() if k in args_name)

    return loss(y_pred, y_gt, **loss_kwargs)


def balance_loss(balanced_loss_list: List[torch.Tensor]) -> torch.Tensor:
    """Balance multiple losses.

    Example:
        loss_a, loss_b, loss_c

        balanced_loss = loss_a * (loss_b * loss_c / (loss_a + loss_b + loss_c))
            + loss_b * (loss_a * loss_c / (loss_a + loss_b + loss_c))
            + loss_c * (loss_a * loss_b / (loss_a + loss_b + loss_c))

    Args:
        balanced_loss_list (List[torch.Tensor]): Loss list.

    Return:
        balanced_loss (torch.Tensor)
    """

    balanced_loss = 0.0
    balanced_sum = sum(balanced_loss_list).detach()
    balanced_product = reduce(lambda x, y: x * y, balanced_loss_list).detach()
    for loss in balanced_loss_list:
        balanced_loss += (loss * balanced_product) / (loss.detach() * balanced_sum)
    return balanced_loss


scan_modules(os.getcwd(), __file__, ['__init__.py', 'builder.py'])

LOSS_REGISTRY.register(nn.L1Loss, 'L1_LOSS')
LOSS_REGISTRY.register(nn.MSELoss, 'L2_LOSS')
LOSS_REGISTRY.register(LogL1Loss, 'LOG_L1_LOSS')
LOSS_REGISTRY.register(LogL2Loss, 'LOG_L2_LOSS')
LOSS_REGISTRY.register(CharbonnierLoss, 'CHARBONNIER_LOSS')
LOSS_REGISTRY.register(EdgeLoss, 'EDGE_LOSS')
LOSS_REGISTRY.register(StyleLoss, 'STYLE_LOSS')
LOSS_REGISTRY.register(PerceptualLoss, 'PERCEPT_LOSS')