"""Evaluation metrics for molecular property prediction."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (binary).
        y_prob: Predicted probabilities.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {}

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"Could not compute PR-AUC: {e}")
        metrics["pr_auc"] = 0.0

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device to run evaluation on.

    Returns:
        Tuple of (metrics_dict, y_true, y_pred, y_prob).
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            data = data.to(device)

            output, _ = model(data)
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    y_true = np.array(all_labels).flatten()
    y_pred = np.array(all_preds).flatten()
    y_prob = np.array(all_probs).flatten()

    metrics = compute_metrics(y_true, y_pred, y_prob)

    return metrics, y_true, y_pred, y_prob


def compute_scaffold_split_metrics(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Dict[str, Any]:
    """Compute metrics specifically for scaffold split evaluation.

    Args:
        model: Model to evaluate.
        data_loader: Data loader with scaffold information.
        device: Device to run evaluation on.

    Returns:
        Dictionary with scaffold-specific metrics.
    """
    model.eval()

    scaffold_preds = {}
    scaffold_labels = {}

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Computing scaffold metrics"):
            data = data.to(device)

            output, _ = model(data)
            probs = torch.sigmoid(output).cpu().numpy()
            labels = data.y.cpu().numpy()

            # Group by scaffold
            scaffolds = None
            if hasattr(data, "scaffold_list"):
                scaffolds = data.scaffold_list
            elif hasattr(data, "scaffold"):
                scaffolds = data.scaffold if isinstance(data.scaffold, list) else [data.scaffold]

            if scaffolds is not None:
                for i, scaffold in enumerate(scaffolds):
                    scaffold_str = str(scaffold)
                    if scaffold_str not in scaffold_preds:
                        scaffold_preds[scaffold_str] = []
                        scaffold_labels[scaffold_str] = []
                    if i < len(probs):
                        scaffold_preds[scaffold_str].append(probs[i])
                    if i < len(labels):
                        scaffold_labels[scaffold_str].append(labels[i])

    # Compute per-scaffold AUC
    scaffold_aucs = {}
    for scaffold, preds in scaffold_preds.items():
        labels = scaffold_labels[scaffold]
        labels_flat = np.array(labels).flatten()
        if len(set(labels_flat.tolist())) > 1:  # Need both classes for AUC
            try:
                preds_flat = np.array(preds).flatten()
                auc = roc_auc_score(labels_flat, preds_flat)
                scaffold_aucs[scaffold] = auc
            except Exception:
                pass

    if scaffold_aucs:
        mean_scaffold_auc = np.mean(list(scaffold_aucs.values()))
        std_scaffold_auc = np.std(list(scaffold_aucs.values()))
    else:
        mean_scaffold_auc = 0.0
        std_scaffold_auc = 0.0

    return {
        "scaffold_split_auc": mean_scaffold_auc,
        "scaffold_split_auc_std": std_scaffold_auc,
        "num_scaffolds": len(scaffold_aucs),
    }


def compute_spectral_alignment_score(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    """Compute spectral alignment score measuring filter quality.

    This metric measures how well the learned spectral filters align with
    the task-relevant frequencies.

    Args:
        model: Model with spectral filters.
        data_loader: Data loader.
        device: Device to run on.

    Returns:
        Spectral alignment score between 0 and 1.
    """
    model.eval()

    all_filter_weights = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            _, auxiliary = model(data)

            if auxiliary["filter_weights"] is not None:
                # Average across layers
                filter_weights = auxiliary["filter_weights"].mean(dim=0)
                all_filter_weights.append(filter_weights.cpu().numpy())

    if not all_filter_weights:
        return 0.0

    # Compute diversity of filter weights (high diversity = good alignment)
    avg_filter_weights = np.mean(all_filter_weights, axis=0)
    entropy = -np.sum(avg_filter_weights * np.log(avg_filter_weights + 1e-8))
    max_entropy = np.log(len(avg_filter_weights))

    # Normalize to [0, 1]
    alignment_score = entropy / max_entropy if max_entropy > 0 else 0.0

    return alignment_score
