"""Analysis and visualization utilities for results."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_training_curves(
    train_losses: List[float], val_aucs: List[float], save_path: Optional[str] = None
) -> None:
    """Plot training loss and validation AUC curves.

    Args:
        train_losses: List of training losses per epoch.
        val_aucs: List of validation AUCs per epoch.
        save_path: Optional path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training loss
    ax1.plot(train_losses, label="Training Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot validation AUC
    ax2.plot(val_aucs, label="Validation AUC", linewidth=2, color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("Validation AUC over Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None
) -> None:
    """Plot ROC curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        save_path: Optional path to save the figure.
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {save_path}")

    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Optional path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close()


def plot_filter_weights(
    filter_weights: np.ndarray, save_path: Optional[str] = None
) -> None:
    """Plot learned spectral filter weights.

    Args:
        filter_weights: Array of shape [num_layers, num_frequencies].
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(12, 6))

    if filter_weights.ndim == 1:
        # Single layer
        plt.bar(range(len(filter_weights)), filter_weights)
        plt.xlabel("Frequency Index")
        plt.ylabel("Filter Weight")
        plt.title("Learned Spectral Filter Weights")
    else:
        # Multiple layers
        for i, weights in enumerate(filter_weights):
            plt.plot(weights, label=f"Layer {i+1}", marker="o", linewidth=2)

        plt.xlabel("Frequency Index")
        plt.ylabel("Filter Weight")
        plt.title("Learned Spectral Filter Weights Across Layers")
        plt.legend()

    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved filter weights plot to {save_path}")

    plt.close()


def generate_analysis_report(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str = "results",
) -> None:
    """Generate comprehensive analysis report with visualizations.

    Args:
        metrics: Dictionary of evaluation metrics.
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        output_dir: Directory to save outputs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot ROC curve
    plot_roc_curve(y_true, y_prob, save_path=output_path / "roc_curve.png")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=output_path / "confusion_matrix.png")

    logger.info(f"Generated analysis report in {output_dir}")
