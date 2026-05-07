"""Continual learning evaluation metrics."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


class ContinualLearningMetrics:
    """Comprehensive metrics for continual learning evaluation.
    
    This class tracks various metrics including accuracy, forgetting,
    forward transfer, and backward transfer across multiple tasks.
    """
    
    def __init__(self, num_tasks: int) -> None:
        """Initialize metrics tracker.
        
        Args:
            num_tasks: Number of tasks in the continual learning scenario
        """
        self.num_tasks = num_tasks
        self.reset()
        
    def reset(self) -> None:
        """Reset all metrics."""
        # Task-specific accuracies: [task_id][eval_task] = accuracy
        self.task_accuracies: List[List[float]] = [[] for _ in range(self.num_tasks)]
        
        # Per-task metrics
        self.task_losses: List[List[float]] = [[] for _ in range(self.num_tasks)]
        
        # Final accuracies after all tasks
        self.final_accuracies: List[float] = [0.0] * self.num_tasks
        
    def update_task_accuracy(
        self,
        task_id: int,
        eval_task: int,
        accuracy: float,
    ) -> None:
        """Update accuracy for a specific task evaluation.
        
        Args:
            task_id: Task being trained
            eval_task: Task being evaluated
            accuracy: Accuracy score
        """
        # Ensure the task_accuracies list is large enough
        while len(self.task_accuracies) <= task_id:
            self.task_accuracies.append([])
            
        if len(self.task_accuracies[task_id]) <= eval_task:
            self.task_accuracies[task_id].extend([0.0] * (eval_task + 1 - len(self.task_accuracies[task_id])))
        self.task_accuracies[task_id][eval_task] = accuracy
        
    def update_task_loss(self, task_id: int, loss: float) -> None:
        """Update loss for a specific task.
        
        Args:
            task_id: Task ID
            loss: Loss value
        """
        # Ensure the task_losses list is large enough
        while len(self.task_losses) <= task_id:
            self.task_losses.append([])
        self.task_losses[task_id].append(loss)
        
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive continual learning metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Average Accuracy (AA)
        metrics["average_accuracy"] = self._compute_average_accuracy()
        
        # Forgetting Measure (FM)
        metrics["forgetting_measure"] = self._compute_forgetting_measure()
        
        # Forward Transfer (FT)
        metrics["forward_transfer"] = self._compute_forward_transfer()
        
        # Backward Transfer (BT)
        metrics["backward_transfer"] = self._compute_backward_transfer()
        
        # Learning Efficiency (LE)
        metrics["learning_efficiency"] = self._compute_learning_efficiency()
        
        # Task-specific metrics
        for task_id in range(self.num_tasks):
            if self.task_accuracies[task_id]:
                metrics[f"task_{task_id}_final_accuracy"] = self.task_accuracies[task_id][task_id]
                
        return metrics
    
    def _compute_average_accuracy(self) -> float:
        """Compute Average Accuracy (AA).
        
        Returns:
            Average accuracy across all tasks
        """
        if not self.task_accuracies:
            return 0.0
            
        # Get final accuracies for each task
        final_accuracies = []
        for task_id in range(self.num_tasks):
            if (self.task_accuracies[task_id] and 
                len(self.task_accuracies[task_id]) > task_id):
                final_accuracies.append(self.task_accuracies[task_id][task_id])
                
        return np.mean(final_accuracies) if final_accuracies else 0.0
    
    def _compute_forgetting_measure(self) -> float:
        """Compute Forgetting Measure (FM).
        
        Returns:
            Average forgetting across all tasks
        """
        if not self.task_accuracies:
            return 0.0
            
        forgetting_values = []
        
        for task_id in range(self.num_tasks):
            if not self.task_accuracies[task_id]:
                continue
                
            # Find peak performance for this task
            peak_acc = 0.0
            for eval_task in range(len(self.task_accuracies[task_id])):
                if eval_task == task_id:  # Only consider evaluation on the same task
                    peak_acc = max(peak_acc, self.task_accuracies[task_id][eval_task])
                    
            # Find final performance
            final_acc = 0.0
            if len(self.task_accuracies[task_id]) > task_id:
                final_acc = self.task_accuracies[task_id][task_id]
                
            forgetting_values.append(peak_acc - final_acc)
            
        return np.mean(forgetting_values) if forgetting_values else 0.0
    
    def _compute_forward_transfer(self) -> float:
        """Compute Forward Transfer (FT).
        
        Returns:
            Average forward transfer across all tasks
        """
        if not self.task_accuracies:
            return 0.0
            
        forward_transfer_values = []
        
        for task_id in range(1, self.num_tasks):  # Skip first task
            if not self.task_accuracies[task_id]:
                continue
                
            # Compare with random baseline (assuming 1/num_classes accuracy)
            baseline_acc = 1.0 / 10  # Assuming 10 classes
            task_acc = 0.0
            
            if len(self.task_accuracies[task_id]) > task_id:
                task_acc = self.task_accuracies[task_id][task_id]
                
            forward_transfer_values.append(task_acc - baseline_acc)
            
        return np.mean(forward_transfer_values) if forward_transfer_values else 0.0
    
    def _compute_backward_transfer(self) -> float:
        """Compute Backward Transfer (BT).
        
        Returns:
            Average backward transfer across all tasks
        """
        if not self.task_accuracies:
            return 0.0
            
        backward_transfer_values = []
        
        for task_id in range(self.num_tasks - 1):  # Skip last task
            if not self.task_accuracies[task_id]:
                continue
                
            # Compare performance before and after learning subsequent tasks
            initial_acc = 0.0
            final_acc = 0.0
            
            if len(self.task_accuracies[task_id]) > task_id:
                initial_acc = self.task_accuracies[task_id][task_id]
                
            # Find final performance after all tasks
            if (self.task_accuracies[-1] and 
                len(self.task_accuracies[-1]) > task_id):
                final_acc = self.task_accuracies[-1][task_id]
                
            backward_transfer_values.append(final_acc - initial_acc)
            
        return np.mean(backward_transfer_values) if backward_transfer_values else 0.0
    
    def _compute_learning_efficiency(self) -> float:
        """Compute Learning Efficiency (LE).
        
        Returns:
            Learning efficiency metric
        """
        if not self.task_accuracies:
            return 0.0
            
        # Simple learning efficiency: final accuracy / number of tasks
        final_accuracies = []
        for task_id in range(self.num_tasks):
            if (self.task_accuracies[task_id] and 
                len(self.task_accuracies[task_id]) > task_id):
                final_accuracies.append(self.task_accuracies[task_id][task_id])
                
        avg_final_acc = np.mean(final_accuracies) if final_accuracies else 0.0
        return avg_final_acc / self.num_tasks
    
    def get_task_matrix(self) -> np.ndarray:
        """Get task accuracy matrix.
        
        Returns:
            Matrix where entry (i,j) is accuracy of task i on task j
        """
        max_tasks = max(len(accs) for accs in self.task_accuracies) if self.task_accuracies else 0
        matrix = np.zeros((self.num_tasks, max_tasks))
        
        for i, accs in enumerate(self.task_accuracies):
            for j, acc in enumerate(accs):
                matrix[i, j] = acc
                
        return matrix
    
    def print_summary(self) -> None:
        """Print a summary of all metrics."""
        metrics = self.compute_metrics()
        
        print("Continual Learning Metrics Summary:")
        print("=" * 40)
        print(f"Average Accuracy (AA): {metrics['average_accuracy']:.4f}")
        print(f"Forgetting Measure (FM): {metrics['forgetting_measure']:.4f}")
        print(f"Forward Transfer (FT): {metrics['forward_transfer']:.4f}")
        print(f"Backward Transfer (BT): {metrics['backward_transfer']:.4f}")
        print(f"Learning Efficiency (LE): {metrics['learning_efficiency']:.4f}")
        print()
        
        print("Task-specific Final Accuracies:")
        for task_id in range(self.num_tasks):
            if f"task_{task_id}_final_accuracy" in metrics:
                print(f"Task {task_id}: {metrics[f'task_{task_id}_final_accuracy']:.4f}")
        
        print("\nTask Accuracy Matrix:")
        matrix = self.get_task_matrix()
        print(matrix)
