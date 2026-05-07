#!/usr/bin/env python3
"""Simple test script to verify the implementation works."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from src.models import SimpleNN, ResNet18Continual
        print("✅ Models imported successfully")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from src.losses import EWCLoss, L2RegularizationLoss, MASLoss, PackNetLoss
        print("✅ Losses imported successfully")
    except Exception as e:
        print(f"❌ Loss import failed: {e}")
        return False
    
    try:
        from src.metrics import ContinualLearningMetrics
        print("✅ Metrics imported successfully")
    except Exception as e:
        print(f"❌ Metrics import failed: {e}")
        return False
    
    try:
        from src.data import ContinualDataLoader, SyntheticContinualDataset
        print("✅ Data modules imported successfully")
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    try:
        from src.train import ContinualTrainer
        print("✅ Trainer imported successfully")
    except Exception as e:
        print(f"❌ Trainer import failed: {e}")
        return False
    
    try:
        from src.utils import set_seed, get_device
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        from src.models import SimpleNN
        from src.data import ContinualDataLoader
        from src.utils import set_seed, get_device
        
        # Set seed
        set_seed(42)
        print("✅ Random seed set")
        
        # Get device
        device = get_device()
        print(f"✅ Device detected: {device}")
        
        # Create model
        model = SimpleNN(input_dim=20, output_dim=2)
        model = model.to(device)
        print("✅ Model created and moved to device")
        
        # Test forward pass
        x = torch.randn(5, 20).to(device)
        output = model(x)
        assert output.shape == (5, 2)
        print("✅ Forward pass works")
        
        # Create data loader
        data_loader = ContinualDataLoader(
            dataset_type="synthetic",
            num_tasks=2,
            batch_size=16,
            num_samples=100,
            input_dim=20,
            num_classes=2
        )
        print("✅ Data loader created")
        
        train_loaders, test_loaders = data_loader.get_all_tasks()
        assert len(train_loaders) == 2
        assert len(test_loaders) == 2
        print("✅ Data loaders created")
        
        # Test data loading
        for data, target in train_loaders[0]:
            assert data.shape[1] == 20
            assert target.shape[0] == data.shape[0]
            break
        print("✅ Data loading works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_training():
    """Test training functionality."""
    print("\nTesting training functionality...")
    
    try:
        import torch
        from src.models import SimpleNN
        from src.data import ContinualDataLoader
        from src.train import ContinualTrainer
        from src.utils import set_seed, get_device
        
        # Set seed
        set_seed(42)
        device = get_device()
        
        # Create small dataset
        data_loader = ContinualDataLoader(
            dataset_type="synthetic",
            num_tasks=2,
            batch_size=16,
            num_samples=50,
            input_dim=10,
            num_classes=2
        )
        
        train_loaders, test_loaders = data_loader.get_all_tasks()
        
        # Create model
        model = SimpleNN(input_dim=10, output_dim=2)
        model = model.to(device)
        
        # Create trainer
        trainer = ContinualTrainer(
            model=model,
            device=device,
            method="ewc",
            method_params={"importance_factor": 100.0}
        )
        print("✅ Trainer created")
        
        # Train on first task
        trainer.train_task(
            task_id=0,
            train_loader=train_loaders[0],
            test_loaders=test_loaders,
            epochs=2,
            verbose=False
        )
        print("✅ Training on first task completed")
        
        # Train on second task
        trainer.train_task(
            task_id=1,
            train_loader=train_loaders[1],
            test_loaders=test_loaders,
            epochs=2,
            verbose=False
        )
        print("✅ Training on second task completed")
        
        # Evaluate
        results = trainer.evaluate_all_tasks(test_loaders, verbose=False)
        print("✅ Evaluation completed")
        
        # Check results
        assert "average_accuracy" in results
        assert 0 <= results["average_accuracy"] <= 1
        print(f"✅ Final accuracy: {results['average_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧠 Lifelong Learning Implementation - Test Suite")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        return False
    
    # Test training
    if not test_training():
        print("\n❌ Training tests failed!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The implementation is working correctly.")
    print("\nYou can now:")
    print("1. Run experiments: python -m src.cli --method ewc")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Run tests: pytest tests/")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
