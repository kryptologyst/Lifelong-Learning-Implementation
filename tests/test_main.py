"""Tests for continual learning implementation."""

import pytest
import torch
import numpy as np
from src.models import SimpleNN, ResNet18Continual
from src.losses import EWCLoss, L2RegularizationLoss, MASLoss, PackNetLoss
from src.metrics import ContinualLearningMetrics
from src.data import SyntheticContinualDataset, DigitsContinualDataset
from src.utils import set_seed, get_device, count_parameters


class TestModels:
    """Test neural network models."""
    
    def test_simple_nn(self):
        """Test SimpleNN model."""
        model = SimpleNN(input_dim=20, output_dim=2)
        
        # Test forward pass
        x = torch.randn(10, 20)
        output = model(x)
        
        assert output.shape == (10, 2)
        assert not torch.isnan(output).any()
        
        # Test feature extraction
        features = model.get_features(x)
        assert features.shape[0] == 10
        
    def test_resnet_continual(self):
        """Test ResNet18Continual model."""
        model = ResNet18Continual(num_classes=5, pretrained=False)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 5)
        assert not torch.isnan(output).any()
        
        # Test feature extraction
        features = model.get_features(x)
        assert features.shape[0] == 2


class TestLosses:
    """Test continual learning loss functions."""
    
    def test_ewc_loss(self):
        """Test EWC loss."""
        model = SimpleNN(input_dim=10, output_dim=2)
        ewc = EWCLoss(importance_factor=1000.0)
        
        # Test loss computation
        task_loss = torch.tensor(1.0)
        total_loss = ewc(model, task_loss)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= task_loss.item()
        
    def test_l2_loss(self):
        """Test L2 regularization loss."""
        model = SimpleNN(input_dim=10, output_dim=2)
        l2_loss = L2RegularizationLoss(lambda_reg=0.01)
        
        # Update reference parameters
        l2_loss.update_reference(model)
        
        # Test loss computation
        task_loss = torch.tensor(1.0)
        total_loss = l2_loss(model, task_loss)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= task_loss.item()
        
    def test_mas_loss(self):
        """Test MAS loss."""
        model = SimpleNN(input_dim=10, output_dim=2)
        mas = MASLoss(importance_factor=1000.0)
        
        # Test loss computation
        task_loss = torch.tensor(1.0)
        total_loss = mas(model, task_loss)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= task_loss.item()
        
    def test_packnet_loss(self):
        """Test PackNet loss."""
        model = SimpleNN(input_dim=10, output_dim=2)
        packnet = PackNetLoss(prune_ratio=0.5)
        
        # Test loss computation
        outputs = torch.randn(5, 2)
        targets = torch.randint(0, 2, (5,))
        task_id = 0
        
        loss = packnet(model, outputs, targets, task_id)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestMetrics:
    """Test continual learning metrics."""
    
    def test_continual_learning_metrics(self):
        """Test ContinualLearningMetrics."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        # Update some metrics
        metrics.update_task_accuracy(0, 0, 0.8)
        metrics.update_task_accuracy(1, 1, 0.9)
        metrics.update_task_accuracy(2, 2, 0.85)
        
        # Compute metrics
        results = metrics.compute_metrics()
        
        assert "average_accuracy" in results
        assert "forgetting_measure" in results
        assert "forward_transfer" in results
        assert "backward_transfer" in results
        
        assert 0 <= results["average_accuracy"] <= 1
        assert results["forgetting_measure"] >= 0


class TestData:
    """Test data loading and preprocessing."""
    
    def test_synthetic_dataset(self):
        """Test SyntheticContinualDataset."""
        dataset = SyntheticContinualDataset(
            task_id=0,
            num_samples=100,
            input_dim=10,
            num_classes=2,
            random_state=42
        )
        
        assert len(dataset) == 100
        
        # Test data loading
        x, y = dataset[0]
        assert x.shape == (10,)
        assert y.shape == ()
        assert 0 <= y.item() <= 1
        
    def test_digits_dataset(self):
        """Test DigitsContinualDataset."""
        dataset = DigitsContinualDataset(
            task_id=0,
            classes_per_task=2,
            random_state=42
        )
        
        assert len(dataset) > 0
        
        # Test data loading
        x, y = dataset[0]
        assert x.shape == (64,)  # Digits are 8x8 = 64 features
        assert y.shape == ()
        assert 0 <= y.item() <= 1  # Remapped to 0, 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
        
    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleNN(input_dim=10, output_dim=2)
        num_params = count_parameters(model)
        
        assert num_params > 0
        assert isinstance(num_params, int)
        
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


@pytest.mark.slow
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_experiment(self):
        """Test end-to-end experiment."""
        from src.data import ContinualDataLoader
        from src.train import ContinualTrainer
        
        # Set seed
        set_seed(42)
        
        # Create data
        data_loader = ContinualDataLoader(
            dataset_type="synthetic",
            num_tasks=2,
            batch_size=16,
            kwargs={"num_samples": 100, "input_dim": 10, "num_classes": 2}
        )
        
        train_loaders, test_loaders = data_loader.get_all_tasks()
        
        # Create model
        model = SimpleNN(input_dim=10, output_dim=2)
        device = get_device()
        model = model.to(device)
        
        # Create trainer
        trainer = ContinualTrainer(
            model=model,
            device=device,
            method="ewc",
            method_params={"importance_factor": 100.0}
        )
        
        # Run short experiment
        results = trainer.run_continual_experiment(
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            epochs_per_task=2,
            verbose=False
        )
        
        # Check results
        assert "average_accuracy" in results
        assert "forgetting_measure" in results
        assert 0 <= results["average_accuracy"] <= 1
        assert results["forgetting_measure"] >= 0
