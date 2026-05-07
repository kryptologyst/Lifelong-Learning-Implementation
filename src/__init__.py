"""Lifelong Learning Implementation - Main Package."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .models import SimpleNN, ResNet18Continual
from .losses import EWCLoss, L2RegularizationLoss, MASLoss, PackNetLoss
from .metrics import ContinualLearningMetrics
from .train import ContinualTrainer
from .data import ContinualDataLoader, SyntheticContinualDataset

__all__ = [
    "SimpleNN",
    "ResNet18Continual", 
    "EWCLoss",
    "L2RegularizationLoss",
    "MASLoss",
    "PackNetLoss",
    "ContinualLearningMetrics",
    "ContinualTrainer",
    "ContinualDataLoader",
    "SyntheticContinualDataset",
]
