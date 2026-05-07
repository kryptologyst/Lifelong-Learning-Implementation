"""Data loading and preprocessing for continual learning."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SyntheticContinualDataset(Dataset):
    """Synthetic dataset for continual learning experiments.
    
    Creates multiple classification tasks with different data distributions
    to simulate continual learning scenarios.
    
    Args:
        task_id: ID of the current task
        num_samples: Number of samples per task
        input_dim: Input feature dimension
        num_classes: Number of classes per task
        noise: Amount of noise in the data
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        task_id: int,
        num_samples: int = 1000,
        input_dim: int = 20,
        num_classes: int = 2,
        noise: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        self.task_id = task_id
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        X, y = make_classification(
            n_samples=num_samples,
            n_features=input_dim,
            n_classes=num_classes,
            n_redundant=0,
            n_informative=input_dim,
            n_clusters_per_class=1,
            random_state=random_state + task_id if random_state else None,
        )
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DigitsContinualDataset(Dataset):
    """Digits dataset adapted for continual learning.
    
    Splits the digits dataset into multiple tasks for continual learning.
    
    Args:
        task_id: ID of the current task
        classes_per_task: Number of classes per task
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        task_id: int,
        classes_per_task: int = 2,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        self.task_id = task_id
        self.classes_per_task = classes_per_task
        
        # Load digits dataset
        digits = load_digits()
        X = digits.data / 16.0  # Normalize to [0, 1]
        y = digits.target
        
        # Determine classes for this task
        start_class = task_id * classes_per_task
        end_class = min(start_class + classes_per_task, 10)
        
        if start_class >= 10:
            raise ValueError(f"Task {task_id} exceeds available classes")
            
        # Filter data for this task
        task_mask = (y >= start_class) & (y < end_class)
        X_task = X[task_mask]
        y_task = y[task_mask] - start_class  # Remap to 0, 1, 2, ...
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_task, y_task, test_size=test_size, random_state=random_state
        )
        
        # Convert to tensors
        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test data for this task."""
        return self.X_test, self.y_test


class ContinualDataLoader:
    """Data loader for continual learning scenarios.
    
    Manages multiple tasks and provides train/test loaders for each task.
    
    Args:
        dataset_type: Type of dataset ('synthetic' or 'digits')
        num_tasks: Number of tasks
        batch_size: Batch size for data loaders
        **kwargs: Additional arguments for dataset creation
    """
    
    def __init__(
        self,
        dataset_type: str = "synthetic",
        num_tasks: int = 5,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        self.dataset_type = dataset_type
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.kwargs = kwargs
        
        self.train_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.train_loaders: List[DataLoader] = []
        self.test_loaders: List[DataLoader] = []
        
        self._create_datasets()
        
    def _create_datasets(self) -> None:
        """Create datasets for all tasks."""
        for task_id in range(self.num_tasks):
            if self.dataset_type == "synthetic":
                # Extract only relevant parameters for synthetic dataset
                synthetic_params = {
                    k: v for k, v in self.kwargs.items() 
                    if k in ['num_samples', 'input_dim', 'num_classes', 'noise', 'random_state']
                }
                dataset = SyntheticContinualDataset(task_id=task_id, **synthetic_params)
                # Create test dataset
                test_dataset = SyntheticContinualDataset(task_id=task_id, **synthetic_params)
                
            elif self.dataset_type == "digits":
                # Extract only relevant parameters for digits dataset
                digits_params = {
                    k: v for k, v in self.kwargs.items() 
                    if k in ['classes_per_task', 'test_size', 'random_state']
                }
                dataset = DigitsContinualDataset(task_id=task_id, **digits_params)
                # Create test dataset from the same task
                test_dataset = DigitsContinualDataset(task_id=task_id, **digits_params)
                
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
                
            self.train_datasets.append(dataset)
            self.test_datasets.append(test_dataset)
            
            # Create data loaders
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
            
            self.train_loaders.append(train_loader)
            self.test_loaders.append(test_loader)
    
    def get_task_data(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        """Get train and test loaders for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} not available")
            
        return self.train_loaders[task_id], self.test_loaders[task_id]
    
    def get_all_tasks(self) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Get all train and test loaders.
        
        Returns:
            Tuple of (train_loaders, test_loaders)
        """
        return self.train_loaders, self.test_loaders
    
    def get_task_info(self, task_id: int) -> Dict[str, Union[int, str]]:
        """Get information about a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task information
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} not available")
            
        dataset = self.train_datasets[task_id]
        
        return {
            "task_id": task_id,
            "dataset_type": self.dataset_type,
            "num_samples": len(dataset),
            "input_dim": dataset.X.shape[1] if hasattr(dataset, 'X') else None,
            "num_classes": len(torch.unique(dataset.y)) if hasattr(dataset, 'y') else None,
        }
