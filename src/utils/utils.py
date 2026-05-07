"""Utility functions for continual learning."""

import random
import numpy as np
import torch
from typing import Optional, Union


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For CUDA deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon) deterministic behavior
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        PyTorch device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size as string (e.g., "1.2M parameters")
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M parameters"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K parameters"
    else:
        return f"{num_params} parameters"


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path: Path to checkpoint
        device: Device to load on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> None:
    """Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    warmup_factor: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        warmup_factor: Warmup factor
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            **{k: v for k, v in kwargs.items() if k != "momentum"}
        )
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif scheduler_name.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
