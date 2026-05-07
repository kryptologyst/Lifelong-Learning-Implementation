"""Streamlit demo for continual learning experiments."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path

from src.models import SimpleNN
from src.data import ContinualDataLoader
from src.train import ContinualTrainer
from src.utils import set_seed, get_device, get_model_size


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Continual Learning Demo",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("🧠 Continual Learning Implementation")
    st.markdown("**Author:** [kryptologyst](https://github.com/kryptologyst)")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Dataset",
        ["synthetic", "digits"],
        help="Choose the dataset type for the experiment"
    )
    
    # Number of tasks
    num_tasks = st.sidebar.slider(
        "Number of Tasks",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of sequential tasks to learn"
    )
    
    # Method selection
    method = st.sidebar.selectbox(
        "Continual Learning Method",
        ["finetune", "l2", "ewc", "mas", "packnet"],
        help="Choose the continual learning algorithm"
    )
    
    # Method parameters
    st.sidebar.subheader("Method Parameters")
    
    if method in ["ewc", "mas"]:
        importance_factor = st.sidebar.slider(
            "Importance Factor",
            min_value=100.0,
            max_value=5000.0,
            value=1000.0,
            step=100.0,
            help="Weight of the regularization term"
        )
        method_params = {"importance_factor": importance_factor}
    elif method == "l2":
        lambda_reg = st.sidebar.slider(
            "Lambda Regularization",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            help="L2 regularization strength"
        )
        method_params = {"lambda_reg": lambda_reg}
    elif method == "packnet":
        prune_ratio = st.sidebar.slider(
            "Prune Ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Fraction of weights to prune per task"
        )
        method_params = {"prune_ratio": prune_ratio}
    else:
        method_params = {}
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs_per_task = st.sidebar.slider(
        "Epochs per Task",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of training epochs per task"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Learning rate for optimization"
    )
    
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=16,
        max_value=128,
        value=32,
        step=16,
        help="Batch size for training"
    )
    
    # Run experiment button
    if st.sidebar.button("🚀 Run Experiment", type="primary"):
        run_experiment(
            dataset_type=dataset_type,
            num_tasks=num_tasks,
            method=method,
            method_params=method_params,
            epochs_per_task=epochs_per_task,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
    
    # Display information
    st.markdown("---")
    st.markdown("### About Continual Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Continual Learning** (also known as lifelong learning) is the ability 
        of a machine learning model to learn new tasks sequentially without 
        forgetting previously learned knowledge.
        
        **Key Challenges:**
        - **Catastrophic Forgetting**: Performance degradation on old tasks
        - **Forward Transfer**: Benefiting from previous tasks
        - **Backward Transfer**: Improving old tasks with new knowledge
        """)
    
    with col2:
        st.markdown("""
        **Methods Implemented:**
        - **Fine-tuning**: Baseline without regularization
        - **L2 Regularization**: Simple weight decay
        - **EWC**: Elastic Weight Consolidation
        - **MAS**: Memory Aware Synapses
        - **PackNet**: Dynamic network pruning
        
        **Metrics:**
        - **Average Accuracy (AA)**: Final performance across tasks
        - **Forgetting Measure (FM)**: How much is forgotten
        - **Forward Transfer (FT)**: Benefit from previous tasks
        - **Backward Transfer (BT)**: Improvement of old tasks
        """)


def run_experiment(
    dataset_type: str,
    num_tasks: int,
    method: str,
    method_params: dict,
    epochs_per_task: int,
    learning_rate: float,
    batch_size: int,
):
    """Run continual learning experiment."""
    
    # Set up progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Set random seed
        set_seed(42)
        
        # Get device
        device = get_device()
        
        # Create data loader
        status_text.text("Creating datasets...")
        progress_bar.progress(0.1)
        
        data_loader = ContinualDataLoader(
            dataset_type=dataset_type,
            num_tasks=num_tasks,
            batch_size=batch_size,
        )
        
        train_loaders, test_loaders = data_loader.get_all_tasks()
        
        # Get task info
        task_info = data_loader.get_task_info(0)
        input_dim = task_info["input_dim"]
        num_classes = task_info["num_classes"]
        
        # Create model
        status_text.text("Creating model...")
        progress_bar.progress(0.2)
        
        model = SimpleNN(
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_dims=[128, 64],
            dropout_rate=0.2,
        )
        
        model = model.to(device)
        
        # Create trainer
        status_text.text("Setting up trainer...")
        progress_bar.progress(0.3)
        
        trainer = ContinualTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            method=method,
            method_params=method_params,
        )
        
        # Display experiment info
        st.markdown("---")
        st.markdown("### Experiment Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset", dataset_type.title())
            st.metric("Tasks", num_tasks)
            st.metric("Method", method.upper())
        
        with col2:
            st.metric("Model Size", get_model_size(model))
            st.metric("Device", str(device))
            st.metric("Epochs/Task", epochs_per_task)
        
        with col3:
            st.metric("Learning Rate", f"{learning_rate:.4f}")
            st.metric("Batch Size", batch_size)
            st.metric("Input Dim", input_dim)
        
        # Run training
        st.markdown("### Training Progress")
        
        # Create placeholders for results
        results_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Track results
        all_results = []
        task_accuracies = []
        
        for task_id in range(num_tasks):
            status_text.text(f"Training Task {task_id + 1}/{num_tasks}...")
            
            # Train on current task
            trainer.train_task(
                task_id=task_id,
                train_loader=train_loaders[task_id],
                test_loaders=test_loaders,
                epochs=epochs_per_task,
                verbose=False,
            )
            
            # Evaluate on all tasks
            task_results = trainer.evaluate_all_tasks(test_loaders, verbose=False)
            all_results.append(task_results)
            
            # Extract task accuracies
            task_accs = []
            for i in range(num_tasks):
                if i < len(trainer.metrics.task_accuracies[task_id]):
                    task_accs.append(trainer.metrics.task_accuracies[task_id][i])
                else:
                    task_accs.append(0.0)
            task_accuracies.append(task_accs)
            
            # Update progress
            progress = 0.3 + (0.7 * (task_id + 1) / num_tasks)
            progress_bar.progress(progress)
            
            # Display intermediate results
            with results_placeholder.container():
                st.markdown(f"**After Task {task_id + 1}:**")
                
                # Create accuracy matrix
                acc_matrix = np.array(task_accuracies)
                
                # Plot accuracy matrix
                fig = px.imshow(
                    acc_matrix,
                    labels=dict(x="Task", y="Training Step", color="Accuracy"),
                    x=[f"Task {i}" for i in range(num_tasks)],
                    y=[f"After Task {i+1}" for i in range(len(task_accuracies))],
                    color_continuous_scale="RdYlBu_r",
                    aspect="auto",
                )
                fig.update_layout(
                    title="Task Accuracy Matrix",
                    xaxis_title="Evaluation Task",
                    yaxis_title="Training Step",
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Final results
        status_text.text("Computing final metrics...")
        progress_bar.progress(1.0)
        
        final_metrics = trainer.metrics.compute_metrics()
        
        # Display final results
        st.markdown("### Final Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average Accuracy",
                f"{final_metrics.get('average_accuracy', 0):.4f}",
                help="Final accuracy averaged across all tasks"
            )
        
        with col2:
            st.metric(
                "Forgetting Measure",
                f"{final_metrics.get('forgetting_measure', 0):.4f}",
                help="How much performance is lost on old tasks",
                delta=f"{final_metrics.get('forgetting_measure', 0):.4f}",
            )
        
        with col3:
            st.metric(
                "Forward Transfer",
                f"{final_metrics.get('forward_transfer', 0):.4f}",
                help="Benefit from previous tasks",
                delta=f"{final_metrics.get('forward_transfer', 0):.4f}",
            )
        
        with col4:
            st.metric(
                "Backward Transfer",
                f"{final_metrics.get('backward_transfer', 0):.4f}",
                help="Improvement of old tasks",
                delta=f"{final_metrics.get('backward_transfer', 0):.4f}",
            )
        
        # Detailed results table
        st.markdown("### Detailed Results")
        
        results_df = pd.DataFrame({
            "Task": [f"Task {i}" for i in range(num_tasks)],
            "Final Accuracy": [final_metrics.get(f"task_{i}_final_accuracy", 0) 
                             for i in range(num_tasks)],
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Learning curves
        st.markdown("### Learning Curves")
        
        # Plot learning curves for each task
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Task Accuracies Over Time"],
        )
        
        for task_id in range(num_tasks):
            task_accs = [acc[task_id] for acc in task_accuracies if len(acc) > task_id]
            if task_accs:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(task_accs))),
                        y=task_accs,
                        mode='lines+markers',
                        name=f'Task {task_id}',
                        line=dict(width=2),
                    ),
                    row=1, col=1,
                )
        
        fig.update_layout(
            title="Task Accuracies Over Training",
            xaxis_title="Training Step",
            yaxis_title="Accuracy",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        status_text.text("✅ Experiment completed successfully!")
        
    except Exception as e:
        st.error(f"Error running experiment: {str(e)}")
        status_text.text("❌ Experiment failed")


if __name__ == "__main__":
    main()
