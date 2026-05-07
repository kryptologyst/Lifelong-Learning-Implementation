#!/usr/bin/env python3
"""Main experiment script for continual learning."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .models import SimpleNN, ResNet18Continual
from .data import ContinualDataLoader
from .train import ContinualTrainer
from .utils import set_seed, get_device, get_model_size


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """Create model based on type.
    
    Args:
        model_type: Type of model ('simple' or 'resnet')
        input_dim: Input dimension
        num_classes: Number of classes
        **kwargs: Additional model arguments
        
    Returns:
        PyTorch model
    """
    if model_type == "simple":
        return SimpleNN(
            input_dim=input_dim,
            output_dim=num_classes,
            **kwargs
        )
    elif model_type == "resnet":
        return ResNet18Continual(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    config: Dict,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run a single continual learning experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory for results
        verbose: Whether to print progress
        
    Returns:
        Experiment results
    """
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Get device
    device = get_device()
    
    # Create data loader
    data_loader = ContinualDataLoader(
        dataset_type=config["data"]["type"],
        num_tasks=config["data"]["num_tasks"],
        batch_size=config["data"]["batch_size"],
        **config["data"].get("kwargs", {})
    )
    
    train_loaders, test_loaders = data_loader.get_all_tasks()
    
    # Get task info
    task_info = data_loader.get_task_info(0)
    input_dim = task_info["input_dim"]
    num_classes = task_info["num_classes"]
    
    # Create model
    model = create_model(
        model_type=config["model"]["type"],
        input_dim=input_dim,
        num_classes=num_classes,
        **config["model"].get("kwargs", {})
    )
    
    model = model.to(device)
    
    if verbose:
        print(f"Model: {get_model_size(model)}")
        print(f"Device: {device}")
        print(f"Number of tasks: {config['data']['num_tasks']}")
        print(f"Method: {config['method']}")
        print("-" * 50)
    
    # Create trainer
    trainer = ContinualTrainer(
        model=model,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        method=config["method"],
        method_params=config.get("method_params", {}),
    )
    
    # Run experiment
    start_time = time.time()
    
    results = trainer.run_continual_experiment(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_per_task=config["training"]["epochs_per_task"],
        verbose=verbose,
    )
    
    end_time = time.time()
    results["total_time"] = end_time - start_time
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    trainer.save_model(output_dir / "model.pt")
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)
    
    if verbose:
        print(f"\nExperiment completed in {results['total_time']:.2f} seconds")
        print(f"Results saved to {output_dir}")
    
    return results


def run_comparison(
    config: Dict,
    output_dir: Path,
    methods: List[str],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run comparison across multiple methods.
    
    Args:
        config: Base experiment configuration
        output_dir: Output directory for results
        methods: List of methods to compare
        verbose: Whether to print progress
        
    Returns:
        Comparison results
    """
    all_results = {}
    
    for method in methods:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment with {method.upper()}")
            print(f"{'='*60}")
        
        # Update config for this method
        method_config = config.copy()
        method_config["method"] = method
        
        # Set method-specific parameters
        if method == "ewc":
            method_config["method_params"] = {"importance_factor": 1000.0}
        elif method == "l2":
            method_config["method_params"] = {"lambda_reg": 0.01}
        elif method == "mas":
            method_config["method_params"] = {"importance_factor": 1000.0}
        elif method == "packnet":
            method_config["method_params"] = {"prune_ratio": 0.5}
        
        # Create method-specific output directory
        method_output_dir = output_dir / method
        
        # Run experiment
        try:
            results = run_experiment(
                config=method_config,
                output_dir=method_output_dir,
                verbose=verbose,
            )
            all_results[method] = results
            
        except Exception as e:
            print(f"Error running {method}: {e}")
            all_results[method] = {"error": str(e)}
    
    # Save comparison results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for method, results in all_results.items():
            if "error" not in results:
                print(f"{method.upper():<10}: AA={results.get('average_accuracy', 0):.4f}, "
                      f"FM={results.get('forgetting_measure', 0):.4f}")
            else:
                print(f"{method.upper():<10}: ERROR - {results['error']}")
    
    return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Continual Learning Experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ewc",
        choices=["ewc", "l2", "mas", "packnet", "finetune"],
        help="Continual learning method"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across all methods"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = OmegaConf.load(config_path)
    else:
        # Default config
        config = OmegaConf.create({
            "seed": 42,
            "data": {
                "type": "synthetic",
                "num_tasks": 5,
                "batch_size": 32,
                "kwargs": {
                    "num_samples": 1000,
                    "input_dim": 20,
                    "num_classes": 2,
                }
            },
            "model": {
                "type": "simple",
                "kwargs": {
                    "hidden_dims": [128, 64],
                    "dropout_rate": 0.2,
                }
            },
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "epochs_per_task": 10,
            },
            "method": args.method,
        })
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Run comparison across all methods
        methods = ["finetune", "l2", "ewc", "mas", "packnet"]
        run_comparison(
            config=config,
            output_dir=output_dir,
            methods=methods,
            verbose=args.verbose,
        )
    else:
        # Run single experiment
        config["method"] = args.method
        run_experiment(
            config=config,
            output_dir=output_dir,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
