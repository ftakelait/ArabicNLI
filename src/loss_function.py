"""
Implementation of focal loss function for NLP Tasks.
Reference: https://arxiv.org/abs/1708.02002
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.utils.one_hot import one_hot


class FocalLoss(nn.Module):
    """
    Implements Focal loss for imbalanced classification tasks.

    Arguments:
        alpha (float): Weighting factor alpha.
        gamma (float, optional): Focusing parameter gamma. Default value is 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output: ‘none’ | ‘mean’ | ‘sum’.

    Shape:
        - Input: (N, C) where C = number of classes, N = Batch Size.
        - Target: (N) where each value is 0 <= targets[i] <= C−1.

    Examples:
        n = 2  # num_classes
        input = torch.randn(16,2)
        target = torch.empty(16, dtype=torch.long).random_(n)
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        output = loss_fn(input, target)
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0, reduction: Optional[str] = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal loss between `input` and `target`.
        
        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
        
        Returns:
            loss (torch.Tensor): The Focal loss between `input` and `target`.
        """

        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not input.shape[0] == target.shape[0]:
            raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device, target.device}")

        # Compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # Create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # Compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")

        return loss


def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float, gamma: Optional[float] = 2.0, reduction: Optional[str] = 'none') -> torch.Tensor:
    """
    Function that computes Focal loss.

    Args:
        input (torch.Tensor): The input tensor.
        target (torch.Tensor): The target tensor.
        alpha (float): Weighting factor alpha.
        gamma (float, optional): Focusing parameter gamma. Default value is 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output: ‘none’ | ‘mean’ | ‘sum’.

    Returns:
        loss (torch.Tensor): The Focal loss between `input` and `target`.
    """

    return FocalLoss(alpha, gamma, reduction)(input, target)
