"""Training utilities for continual learning."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..models import SimpleNN, ResNet18Continual
from ..losses import EWCLoss, L2RegularizationLoss, MASLoss, PackNetLoss
from ..metrics import ContinualLearningMetrics
from ..utils import set_seed, get_device


class ContinualTrainer:
    """Trainer for continual learning experiments.
    
    Supports multiple continual learning algorithms including EWC, L2 regularization,
    MAS, and PackNet.
    
    Args:
        model: Neural network model
        device: Device to run training on
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        method: Continual learning method ('ewc', 'l2', 'mas', 'packnet', 'finetune')
        method_params: Parameters specific to the chosen method
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        method: str = "ewc",
        method_params: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.method = method
        self.method_params = method_params or {}
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Initialize continual learning method
        self.cl_method = self._initialize_cl_method()
        
        # Metrics tracker
        self.metrics = ContinualLearningMetrics(num_tasks=0)
        
    def _initialize_cl_method(self) -> Optional[nn.Module]:
        """Initialize the continual learning method.
        
        Returns:
            Continual learning method instance
        """
        if self.method == "ewc":
            return EWCLoss(
                importance_factor=self.method_params.get("importance_factor", 1000.0)
            )
        elif self.method == "l2":
            return L2RegularizationLoss(
                lambda_reg=self.method_params.get("lambda_reg", 0.01)
            )
        elif self.method == "mas":
            return MASLoss(
                importance_factor=self.method_params.get("importance_factor", 1000.0)
            )
        elif self.method == "packnet":
            return PackNetLoss(
                prune_ratio=self.method_params.get("prune_ratio", 0.5)
            )
        elif self.method == "finetune":
            return None  # No regularization
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        test_loaders: List[DataLoader],
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train on a single task.
        
        Args:
            task_id: ID of the current task
            train_loader: Training data loader
            test_loaders: List of test loaders for all tasks
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        # Update metrics tracker
        if task_id >= self.metrics.num_tasks:
            self.metrics.num_tasks = task_id + 1
        
        # Handle PackNet pruning
        if self.method == "packnet" and self.cl_method is not None:
            self.cl_method.prune_for_task(
                self.model, task_id, train_loader, self.device
            )
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            if verbose:
                pbar = tqdm(train_loader, desc=f"Task {task_id}, Epoch {epoch+1}")
            else:
                pbar = train_loader
                
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Compute task loss
                task_loss = nn.CrossEntropyLoss()(output, target)
                
                # Add continual learning regularization
                if self.cl_method is not None:
                    if self.method == "packnet":
                        total_loss = self.cl_method(
                            self.model, output, target, task_id
                        )
                    else:
                        total_loss = self.cl_method(self.model, task_loss)
                else:
                    total_loss = task_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                if verbose and batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{epoch_loss / num_batches:.4f}',
                        'task_loss': f'{task_loss.item():.4f}'
                    })
            
            # Update metrics
            avg_loss = epoch_loss / num_batches
            self.metrics.update_task_loss(task_id, avg_loss)
            
            if verbose:
                print(f"Task {task_id}, Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Evaluate on all tasks
        task_metrics = self.evaluate_all_tasks(test_loaders, verbose=verbose)
        
        # Update continual learning method after task completion
        if self.cl_method is not None:
            if self.method == "ewc":
                self.cl_method.update_fisher_info(
                    self.model, train_loader, self.device
                )
            elif self.method == "mas":
                self.cl_method.update_importance(
                    self.model, train_loader, self.device
                )
            elif self.method == "l2":
                self.cl_method.update_reference(self.model)
        
        return task_metrics
    
    def evaluate_all_tasks(
        self,
        test_loaders: List[DataLoader],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model on all tasks.
        
        Args:
            test_loaders: List of test loaders for all tasks
            verbose: Whether to print evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            for task_id, test_loader in enumerate(test_loaders):
                correct = 0
                total = 0
                
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, predicted = torch.max(output, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                
                accuracy = correct / total
                self.metrics.update_task_accuracy(task_id, task_id, accuracy)
                
                if verbose:
                    print(f"Task {task_id} Accuracy: {accuracy:.4f}")
        
        return self.metrics.compute_metrics()
    
    def evaluate_single_task(
        self,
        test_loader: DataLoader,
        task_id: int,
    ) -> float:
        """Evaluate model on a single task.
        
        Args:
            test_loader: Test data loader
            task_id: ID of the task
            
        Returns:
            Accuracy on the task
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def run_continual_experiment(
        self,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        epochs_per_task: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Run complete continual learning experiment.
        
        Args:
            train_loaders: List of training loaders for all tasks
            test_loaders: List of test loaders for all tasks
            epochs_per_task: Number of epochs per task
            verbose: Whether to print progress
            
        Returns:
            Final metrics dictionary
        """
        if verbose:
            print(f"Starting continual learning experiment with {self.method}")
            print(f"Number of tasks: {len(train_loaders)}")
            print(f"Epochs per task: {epochs_per_task}")
            print("-" * 50)
        
        # Train on each task sequentially
        for task_id in range(len(train_loaders)):
            if verbose:
                print(f"\nTraining Task {task_id + 1}/{len(train_loaders)}")
            
            # Train on current task
            self.train_task(
                task_id=task_id,
                train_loader=train_loaders[task_id],
                test_loaders=test_loaders,
                epochs=epochs_per_task,
                verbose=verbose,
            )
            
            # Evaluate on all tasks after training
            if verbose:
                print(f"\nEvaluation after Task {task_id + 1}:")
                self.evaluate_all_tasks(test_loaders, verbose=True)
        
        # Final evaluation
        if verbose:
            print("\n" + "=" * 50)
            print("FINAL RESULTS")
            print("=" * 50)
            self.metrics.print_summary()
        
        return self.metrics.compute_metrics()
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'method': self.method,
            'method_params': self.method_params,
            'metrics': self.metrics,
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.method = checkpoint['method']
        self.method_params = checkpoint['method_params']
        self.metrics = checkpoint['metrics']
