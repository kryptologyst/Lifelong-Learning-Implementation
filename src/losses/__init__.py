"""Continual learning loss functions."""

from .losses import EWCLoss, L2RegularizationLoss, MASLoss, PackNetLoss

__all__ = ["EWCLoss", "L2RegularizationLoss", "MASLoss", "PackNetLoss"]
