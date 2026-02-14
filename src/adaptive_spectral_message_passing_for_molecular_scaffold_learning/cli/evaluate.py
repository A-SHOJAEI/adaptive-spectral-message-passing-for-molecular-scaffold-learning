#!/usr/bin/env python
"""Evaluation script for trained models."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.loader import (
    get_data_loaders,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.evaluation.analysis import (
    generate_analysis_report,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.evaluation.metrics import (
    compute_scaffold_split_metrics,
    compute_spectral_alignment_score,
    evaluate_model,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import (
    AdaptiveSpectralGNN,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.utils.config import (
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
    parser = argparse.ArgumentParser(description="Evaluate trained molecular GNN model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for output files"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting evaluation pipeline...")

    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = checkpoint.get("config", {})
            logger.info("Using configuration from checkpoint")

        # Set random seeds
        seed = config.get("seed", 42)
        set_seed(seed)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(config, add_spectral=True)

        # Select split
        if args.split == "train":
            data_loader = train_loader
        elif args.split == "val":
            data_loader = val_loader
        else:
            data_loader = test_loader

        logger.info(f"Evaluating on {args.split} set with {len(data_loader.dataset)} samples")

        # Create model
        logger.info("Creating model...")
        model = AdaptiveSpectralGNN.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Evaluate model
        logger.info("Computing metrics...")
        metrics, y_true, y_pred, y_prob = evaluate_model(model, data_loader, device)

        # Compute scaffold-specific metrics
        scaffold_metrics = compute_scaffold_split_metrics(model, data_loader, device)
        metrics.update(scaffold_metrics)

        # Compute spectral alignment score
        spectral_score = compute_spectral_alignment_score(model, data_loader, device)
        metrics["spectral_alignment_score"] = spectral_score

        # Print metrics table
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)

        metrics_table = []
        for metric_name, value in sorted(metrics.items()):
            logger.info(f"{metric_name:30s}: {value:.4f}")
            metrics_table.append({"metric": metric_name, "value": value})

        logger.info("=" * 60)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_json_path = output_dir / f"{args.split}_metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_json_path}")

        # Save metrics as CSV
        metrics_csv_path = output_dir / f"{args.split}_metrics.csv"
        pd.DataFrame(metrics_table).to_csv(metrics_csv_path, index=False)
        logger.info(f"Saved metrics to {metrics_csv_path}")

        # Save predictions
        predictions_path = output_dir / f"{args.split}_predictions.csv"
        predictions_df = pd.DataFrame(
            {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
        )
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")

        # Generate analysis report with visualizations
        logger.info("Generating analysis report...")
        generate_analysis_report(metrics, y_true, y_pred, y_prob, output_dir=str(output_dir))

        logger.info(f"Evaluation completed! Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
