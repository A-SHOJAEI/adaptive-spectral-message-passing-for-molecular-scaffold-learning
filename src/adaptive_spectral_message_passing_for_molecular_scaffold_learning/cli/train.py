#!/usr/bin/env python
"""Training script for adaptive spectral message passing model."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.loader import (
    get_data_loaders,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.evaluation.analysis import (
    plot_training_curves,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import (
    AdaptiveSpectralGNN,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.training.trainer import (
    Trainer,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.utils.config import (
    create_directories,
    load_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train adaptive spectral GNN for molecular property prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting training pipeline...")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Set random seeds
        seed = config.get("seed", 42)
        set_seed(seed)

        # Create necessary directories
        create_directories(config)

        # Set device
        device_name = config.get("device", "cuda")
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(config, add_spectral=True)

        # Create model
        logger.info("Creating model...")
        model = AdaptiveSpectralGNN.from_config(config)
        logger.info(
            f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Load checkpoint if provided
        start_epoch: int = 0
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1

        # Create trainer
        use_amp = config.get("mixed_precision", True) and device.type == "cuda"
        trainer = Trainer(model, config, device, use_amp=use_amp)

        # Initialize MLflow tracking (optional)
        try:
            import mlflow

            mlflow.set_experiment("adaptive_spectral_gnn")
            mlflow.start_run()
            mlflow.log_params({
                "hidden_dim": config.get("model", {}).get("hidden_dim", 128),
                "num_layers": config.get("model", {}).get("num_layers", 4),
                "learning_rate": config.get("training", {}).get("learning_rate", 0.001),
                "batch_size": config.get("training", {}).get("batch_size", 32),
                "seed": seed,
            })
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")

        # Train model
        logger.info("Starting training...")
        checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints")

        try:
            history = trainer.train(train_loader, val_loader, checkpoint_dir)
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        # Save training history
        results_dir = config.get("logging", {}).get("results_dir", "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        history_path = Path(results_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Plot training curves
        plot_path = Path(results_dir) / "training_curves.png"
        plot_training_curves(
            history["train_loss"], history["val_auc"], save_path=str(plot_path)
        )

        # Log to MLflow
        try:
            import mlflow

            mlflow.log_metric("best_val_auc", trainer.best_val_auc)
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(plot_path))
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        logger.info(f"Training completed! Best validation AUC: {trainer.best_val_auc:.4f}")

        # Evaluate on test set
        from adaptive_spectral_message_passing_for_molecular_scaffold_learning.evaluation.metrics import (
            evaluate_model,
        )

        logger.info("Evaluating on test set...")
        test_metrics, _, _, _ = evaluate_model(model, test_loader, device)

        logger.info("Test set metrics:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Save test metrics
        test_metrics_path = Path(results_dir) / "test_metrics.json"
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test metrics to {test_metrics_path}")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
