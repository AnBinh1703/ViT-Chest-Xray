"""
Evaluation Utilities for Chest X-ray Classification
===================================================

This module contains evaluation functions, metrics calculation,
and visualization tools for model performance analysis.

Features:
- AUC calculation for multi-label classification
- Confusion matrix generation
- Performance visualization
- Cross-validation utilities
- Model comparison tools
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def calculate_auc_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate AUC for multi-label classification.

    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)

    Returns:
        Tuple of (mean_auc, individual_aucs)
    """
    num_classes = y_true.shape[1]
    individual_aucs = []

    for i in range(num_classes):
        try:
            # Only calculate AUC if class has both positive and negative samples
            if len(np.unique(y_true[:, i])) > 1:
                auc_score = roc_auc_score(y_true[:, i], y_pred[:, i])
                individual_aucs.append(auc_score)
            else:
                individual_aucs.append(0.5)  # Default AUC for single-class
        except ValueError:
            individual_aucs.append(0.5)

    mean_auc = np.mean(individual_aucs)
    return mean_auc, np.array(individual_aucs)


def plot_roc_curves(y_true: np.ndarray, y_pred: np.ndarray,
                   class_names: List[str], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for all classes.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for All Classes')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix_multilabel(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: List[str], threshold: float = 0.5,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrices for multi-label classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        class_names: List of class names
        threshold: Classification threshold
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calculate confusion matrices for each class
    n_classes = len(class_names)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()

    for i, class_name in enumerate(class_names):
        if i >= len(axes):
            break

        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[i].set_title(f'{class_name}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')

    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Confusion Matrices for All Classes', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                  device: torch.device, class_names: List[str],
                  save_dir: Optional[str] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive model evaluation.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: List of class names
        save_dir: Directory to save evaluation plots

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all predictions and labels
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    mean_auc, individual_aucs = calculate_auc_multilabel(y_true, y_pred)

    # Create results dictionary
    results = {
        'mean_auc': mean_auc,
        'individual_aucs': individual_aucs,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names
    }

    # Generate plots if save_dir is provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # ROC curves
        plot_roc_curves(y_true, y_pred, class_names,
                       save_path=os.path.join(save_dir, 'roc_curves.png'))

        # Confusion matrices
        plot_confusion_matrix_multilabel(y_true, y_pred, class_names,
                                       save_path=os.path.join(save_dir, 'confusion_matrices.png'))

        # AUC bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(class_names)), individual_aucs, color='skyblue')
        ax.set_xlabel('Disease Classes')
        ax.set_ylabel('AUC Score')
        ax.set_title('AUC Scores by Disease Class')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        # Add value labels on bars
        for bar, auc_val in zip(bars, individual_aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '.3f', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'auc_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return results


def compare_models(model_results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple models based on their evaluation results.

    Args:
        model_results: Dictionary with model names as keys and evaluation results as values
        save_path: Path to save the comparison plot

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    model_names = list(model_results.keys())
    mean_aucs = [results['mean_auc'] for results in model_results.values()]

    # Mean AUC comparison
    bars = ax1.bar(model_names, mean_aucs, color=['lightcoral', 'lightgreen', 'lightblue', 'gold'])
    ax1.set_ylabel('Mean AUC Score')
    ax1.set_title('Mean AUC Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 0.95)

    # Add value labels
    for bar, auc_val in zip(bars, mean_aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                '.3f', ha='center', va='bottom', fontweight='bold')

    # Individual class AUCs
    class_names = model_results[model_names[0]]['class_names']
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)

    for i, (model_name, results) in enumerate(model_results.items()):
        offset = (i - len(model_names)/2 + 0.5) * width
        ax2.bar(x + offset, results['individual_aucs'], width,
               label=model_name, alpha=0.8)

    ax2.set_xlabel('Disease Classes')
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC by Class Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)

    plt.suptitle('Model Performance Comparison', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def calculate_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: Binary label matrix (N, num_classes)

    Returns:
        Array of class weights
    """
    num_samples, num_classes = labels.shape
    class_weights = []

    for i in range(num_classes):
        pos_count = np.sum(labels[:, i])
        neg_count = num_samples - pos_count

        # Use inverse frequency weighting
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0

        class_weights.append(weight)

    return np.array(class_weights)


def bootstrap_auc_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                    n_bootstraps: int = 1000, alpha: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate confidence interval for AUC using bootstrapping.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bootstraps: Number of bootstrap samples
        alpha: Confidence level

    Returns:
        Tuple of (mean_auc, (lower_ci, upper_ci))
    """
    auc_scores = []

    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate AUC
        try:
            auc_score, _ = calculate_auc_multilabel(y_true_boot, y_pred_boot)
            auc_scores.append(auc_score)
        except:
            continue

    auc_scores = np.array(auc_scores)
    mean_auc = np.mean(auc_scores)

    # Calculate confidence interval
    lower_percentile = (1 - alpha) / 2 * 100
    upper_percentile = (1 + alpha) / 2 * 100

    lower_ci = np.percentile(auc_scores, lower_percentile)
    upper_ci = np.percentile(auc_scores, upper_percentile)

    return mean_auc, (lower_ci, upper_ci)