#!/usr/bin/env python
"""Prediction script for inference on new molecules."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.loader import (
    mol_to_graph,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.preprocessing import (
    add_spectral_features_to_data,
    compute_scaffold_complexity,
    generate_scaffold,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import (
    AdaptiveSpectralGNN,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.utils.config import (
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run predictions on new molecules")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        nargs="+",
        help="SMILES strings to predict (space-separated)",
    )
    parser.add_argument(
        "--smiles-file", type=str, help="Path to file containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save predictions (JSON format)"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def load_smiles_from_file(file_path: str) -> List[str]:
    """Load SMILES strings from file.

    Args:
        file_path: Path to file containing SMILES.

    Returns:
        List of SMILES strings.
    """
    with open(file_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list


def predict_molecule(
    model: torch.nn.Module,
    smiles: str,
    device: torch.device,
    num_frequencies: int = 16,
) -> Tuple[float, int, Dict[str, any]]:
    """Predict property for a single molecule.

    Args:
        model: Trained model.
        smiles: SMILES string of molecule.
        device: Device to run prediction on.
        num_frequencies: Number of spectral frequencies.

    Returns:
        Tuple of (probability, predicted_class, metadata).
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {smiles}")
            return 0.0, 0, {"error": "Invalid SMILES"}

        # Convert to graph
        data = mol_to_graph(mol)

        # Add scaffold information
        scaffold = generate_scaffold(smiles)
        complexity = compute_scaffold_complexity(mol)
        data.scaffold = scaffold
        data.complexity = torch.tensor([complexity], dtype=torch.float)

        # Add spectral features
        data = add_spectral_features_to_data(data, num_frequencies)

        # Create a batch of size 1 for model input
        batch = Batch.from_data_list([data]).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            output, _ = model(batch)
            probability = torch.sigmoid(output).item()
            predicted_class = int(probability > 0.5)

        metadata = {
            "scaffold": scaffold,
            "complexity": complexity,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
        }

        return probability, predicted_class, metadata

    except Exception as e:
        logger.error(f"Prediction failed for {smiles}: {e}")
        return 0.0, 0, {"error": str(e)}


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting prediction pipeline...")

    try:
        # Load SMILES
        if args.smiles:
            smiles_list = args.smiles
        elif args.smiles_file:
            smiles_list = load_smiles_from_file(args.smiles_file)
            logger.info(f"Loaded {len(smiles_list)} SMILES from {args.smiles_file}")
        else:
            logger.error("Must provide either --smiles or --smiles-file")
            sys.exit(1)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create model
        logger.info("Creating model...")
        model = AdaptiveSpectralGNN.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        num_frequencies = config.get("spectral", {}).get("num_frequencies", 16)

        # Run predictions
        logger.info(f"Running predictions on {len(smiles_list)} molecules...")
        predictions = []

        for smiles in smiles_list:
            prob, pred_class, metadata = predict_molecule(
                model, smiles, device, num_frequencies
            )

            result = {
                "smiles": smiles,
                "probability": prob,
                "predicted_class": pred_class,
                "confidence": max(prob, 1 - prob),
                "metadata": metadata,
            }

            predictions.append(result)

            # Print result
            logger.info(
                f"SMILES: {smiles[:50]}... | "
                f"Class: {pred_class} | "
                f"Probability: {prob:.4f} | "
                f"Confidence: {max(prob, 1 - prob):.4f}"
            )

        # Save predictions
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Saved predictions to {args.output}")
        else:
            # Print as JSON to stdout
            print(json.dumps(predictions, indent=2))

        logger.info(f"Prediction completed for {len(smiles_list)} molecules!")

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
