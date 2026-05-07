#!/usr/bin/env python3
"""Simple example demonstrating continual learning with EWC."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import SimpleNN
from src.data import ContinualDataLoader
from src.train import ContinualTrainer
from src.utils import set_seed, get_device


def main():
    """Run a simple continual learning experiment."""
    print("🧠 Lifelong Learning Implementation - Simple Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loader with 3 tasks
    print("\nCreating datasets...")
    data_loader = ContinualDataLoader(
        dataset_type="synthetic",
        num_tasks=3,
        batch_size=32,
        num_samples=500,
        input_dim=20,
        num_classes=2
    )
    
    train_loaders, test_loaders = data_loader.get_all_tasks()
    print(f"Created {len(train_loaders)} tasks")
    
    # Create model
    print("\nCreating model...")
    model = SimpleNN(input_dim=20, output_dim=2, hidden_dims=[64, 32])
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer with EWC
    print("\nSetting up EWC trainer...")
    trainer = ContinualTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        method="ewc",
        method_params={"importance_factor": 1000.0}
    )
    
    # Run continual learning experiment
    print("\nStarting continual learning experiment...")
    print("-" * 40)
    
    results = trainer.run_continual_experiment(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_per_task=5,
        verbose=True
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Average Accuracy: {results['average_accuracy']:.4f}")
    print(f"Forgetting Measure: {results['forgetting_measure']:.4f}")
    print(f"Forward Transfer: {results['forward_transfer']:.4f}")
    print(f"Backward Transfer: {results['backward_transfer']:.4f}")
    print(f"Learning Efficiency: {results['learning_efficiency']:.4f}")
    
    print("\nTask-specific accuracies:")
    for i in range(3):
        if f"task_{i}_final_accuracy" in results:
            print(f"  Task {i}: {results[f'task_{i}_final_accuracy']:.4f}")
    
    print("\n✅ Experiment completed successfully!")
    print("\nTo run more experiments:")
    print("1. Try different methods: python -m src.cli --method mas")
    print("2. Run comparison: python -m src.cli --compare")
    print("3. Launch interactive demo: streamlit run demo/app.py")


if __name__ == "__main__":
    main()
