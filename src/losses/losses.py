"""Continual learning loss functions."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EWCLoss(nn.Module):
    """Elastic Weight Consolidation (EWC) loss for continual learning.
    
    EWC prevents catastrophic forgetting by penalizing changes to important
    weights based on Fisher Information Matrix diagonal approximation.
    
    Args:
        importance_factor: Weight of the EWC regularization term
        fisher_diagonal: Diagonal of Fisher Information Matrix
        optimal_params: Optimal parameters from previous tasks
    """
    
    def __init__(
        self,
        importance_factor: float = 1000.0,
        fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None,
        optimal_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.importance_factor = importance_factor
        self.fisher_diagonal = fisher_diagonal or {}
        self.optimal_params = optimal_params or {}
        
    def forward(
        self,
        model: nn.Module,
        task_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute EWC loss.
        
        Args:
            model: Neural network model
            task_loss: Task-specific loss
            
        Returns:
            Total loss including EWC regularization
        """
        ewc_loss = torch.tensor(0.0, device=task_loss.device)
        
        for name, param in model.named_parameters():
            if name in self.fisher_diagonal and name in self.optimal_params:
                fisher = self.fisher_diagonal[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
                
        return task_loss + self.importance_factor * ewc_loss
    
    def update_fisher_info(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Update Fisher Information Matrix diagonal.
        
        Args:
            model: Neural network model
            dataloader: Data loader for computing Fisher information
            device: Device to run computation on
        """
        model.eval()
        fisher_diagonal = {}
        
        # Initialize Fisher diagonal
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_diagonal[name] = torch.zeros_like(param)
                
        # Compute Fisher Information
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_diagonal[name] += param.grad ** 2
                    
        # Average over batches
        for name in fisher_diagonal:
            fisher_diagonal[name] /= len(dataloader)
            
        self.fisher_diagonal = fisher_diagonal
        
        # Save current parameters as optimal
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }


class L2RegularizationLoss(nn.Module):
    """L2 regularization loss for continual learning.
    
    Simple baseline that penalizes large parameter changes.
    
    Args:
        lambda_reg: Regularization strength
        reference_params: Reference parameters to regularize against
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        reference_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.reference_params = reference_params or {}
        
    def forward(
        self,
        model: nn.Module,
        task_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2 regularized loss.
        
        Args:
            model: Neural network model
            task_loss: Task-specific loss
            
        Returns:
            Total loss including L2 regularization
        """
        l2_loss = torch.tensor(0.0, device=task_loss.device)
        
        for name, param in model.named_parameters():
            if name in self.reference_params:
                ref_param = self.reference_params[name]
                l2_loss += ((param - ref_param) ** 2).sum()
                
        return task_loss + self.lambda_reg * l2_loss
    
    def update_reference(self, model: nn.Module) -> None:
        """Update reference parameters.
        
        Args:
            model: Neural network model
        """
        self.reference_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }


class MASLoss(nn.Module):
    """Memory Aware Synapses (MAS) loss for continual learning.
    
    MAS computes importance weights based on the sensitivity of the output
    function to parameter changes.
    
    Args:
        importance_factor: Weight of the MAS regularization term
        importance_weights: Importance weights for each parameter
        reference_params: Reference parameters
    """
    
    def __init__(
        self,
        importance_factor: float = 1000.0,
        importance_weights: Optional[Dict[str, torch.Tensor]] = None,
        reference_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.importance_factor = importance_factor
        self.importance_weights = importance_weights or {}
        self.reference_params = reference_params or {}
        
    def forward(
        self,
        model: nn.Module,
        task_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MAS loss.
        
        Args:
            model: Neural network model
            task_loss: Task-specific loss
            
        Returns:
            Total loss including MAS regularization
        """
        mas_loss = torch.tensor(0.0, device=task_loss.device)
        
        for name, param in model.named_parameters():
            if name in self.importance_weights and name in self.reference_params:
                importance = self.importance_weights[name]
                ref_param = self.reference_params[name]
                mas_loss += (importance * (param - ref_param) ** 2).sum()
                
        return task_loss + self.importance_factor * mas_loss
    
    def update_importance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Update importance weights using MAS.
        
        Args:
            model: Neural network model
            dataloader: Data loader for computing importance
            device: Device to run computation on
        """
        model.eval()
        importance_weights = {}
        
        # Initialize importance weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                importance_weights[name] = torch.zeros_like(param)
                
        # Compute importance weights
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            
            model.zero_grad()
            output = model(data)
            
            # Compute gradient of output magnitude w.r.t. parameters
            output_magnitude = torch.sum(output ** 2)
            output_magnitude.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    importance_weights[name] += torch.abs(param.grad)
                    
        # Average over batches
        for name in importance_weights:
            importance_weights[name] /= len(dataloader)
            
        self.importance_weights = importance_weights
        
        # Save current parameters as reference
        self.reference_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }


class PackNetLoss(nn.Module):
    """PackNet loss with dynamic masking.
    
    PackNet uses dynamic pruning to allocate different subnetworks
    for different tasks.
    
    Args:
        base_loss: Base loss function (e.g., CrossEntropyLoss)
        prune_ratio: Fraction of weights to prune per task
    """
    
    def __init__(
        self,
        base_loss: nn.Module = nn.CrossEntropyLoss(),
        prune_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.prune_ratio = prune_ratio
        self.masks: Dict[int, Dict[str, torch.Tensor]] = {}
        
    def forward(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """Compute PackNet loss.
        
        Args:
            model: Neural network model
            outputs: Model outputs
            targets: Target labels
            task_id: Current task ID
            
        Returns:
            Task loss
        """
        return self.base_loss(outputs, targets)
    
    def prune_for_task(
        self,
        model: nn.Module,
        task_id: int,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Prune weights for a new task.
        
        Args:
            model: Neural network model
            task_id: ID of the new task
            dataloader: Data loader for importance estimation
            device: Device to run computation on
        """
        mask = {}
        
        # Compute importance scores
        importance_scores = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                importance_scores[name] = torch.abs(param.data)
                
        # Create masks based on importance
        for name, scores in importance_scores.items():
            flat_scores = scores.flatten()
            num_prune = int(len(flat_scores) * self.prune_ratio)
            
            # Keep top-k weights
            _, indices = torch.topk(flat_scores, len(flat_scores) - num_prune)
            mask[name] = torch.zeros_like(flat_scores)
            mask[name][indices] = 1
            mask[name] = mask[name].reshape(scores.shape)
            
        self.masks[task_id] = mask
        self._apply_mask(model, task_id)
    
    def _apply_mask(self, model: nn.Module, task_id: int) -> None:
        """Apply mask for specific task.
        
        Args:
            model: Neural network model
            task_id: Task ID
        """
        if task_id in self.masks:
            mask = self.masks[task_id]
            for name, param in model.named_parameters():
                if name in mask:
                    param.data *= mask[name]
                    if param.grad is not None:
                        param.grad *= mask[name]
    
    def switch_to_task(self, model: nn.Module, task_id: int) -> None:
        """Switch to a specific task's mask.
        
        Args:
            model: Neural network model
            task_id: Task ID
        """
        self._apply_mask(model, task_id)
