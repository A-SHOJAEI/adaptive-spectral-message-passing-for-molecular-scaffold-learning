"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root and src/ to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data


@pytest.fixture
def device() -> torch.device:
    """Get device for testing.

    Returns:
        CPU device for reproducible tests.
    """
    return torch.device("cpu")


@pytest.fixture
def sample_smiles() -> str:
    """Get a sample SMILES string.

    Returns:
        SMILES string for benzene.
    """
    return "c1ccccc1"


@pytest.fixture
def sample_molecule(sample_smiles: str) -> Chem.Mol:
    """Create a sample RDKit molecule.

    Args:
        sample_smiles: SMILES string fixture.

    Returns:
        RDKit molecule object.
    """
    return Chem.MolFromSmiles(sample_smiles)


@pytest.fixture
def sample_graph_data() -> Data:
    """Create a sample PyG Data object.

    Returns:
        Sample graph data for testing.
    """
    # Simple graph: triangle (3 nodes, 3 edges)
    # Use 6 features to match molecular graph atom features
    x = torch.tensor(
        [[6.0, 2.0, 0.0, 0.0, 1.0, 1.0],  # Carbon in aromatic ring
         [6.0, 2.0, 0.0, 0.0, 1.0, 1.0],  # Carbon in aromatic ring
         [6.0, 2.0, 0.0, 0.0, 1.0, 1.0]], # Carbon in aromatic ring
        dtype=torch.float
    )

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)

    edge_attr = torch.tensor(
        [[1.5, 1.0, 1.0], [1.5, 1.0, 1.0], [1.5, 1.0, 1.0],
         [1.5, 1.0, 1.0], [1.5, 1.0, 1.0], [1.5, 1.0, 1.0]],
        dtype=torch.float
    )

    y = torch.tensor([1.0], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration dictionary.

    Returns:
        Configuration dictionary for testing.
    """
    return {
        "model": {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_spectral_filters": 4,
            "dropout": 0.1,
            "spectral_dropout": 0.1,
            "use_batch_norm": True,
            "readout": "mean",
        },
        "spectral": {
            "num_frequencies": 8,
            "adaptive_filter": True,
            "normalize_laplacian": True,
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "grad_clip": 1.0,
            "early_stopping_patience": 5,
            "min_delta": 0.0001,
        },
        "curriculum": {"enabled": False},
        "optimizer": {"type": "adam", "betas": [0.9, 0.999]},
        "scheduler": {"type": "cosine", "T_max": 10},
        "regularization": {
            "spectral_smoothness_weight": 0.01,
            "scaffold_consistency_weight": 0.01,
            "filter_diversity_weight": 0.001,
        },
        "seed": 42,
    }
