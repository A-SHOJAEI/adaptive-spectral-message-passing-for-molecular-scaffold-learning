"""Tests for data loading and preprocessing."""

import pytest
import torch
from rdkit import Chem

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.loader import (
    mol_to_graph,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.preprocessing import (
    compute_graph_laplacian,
    compute_scaffold_complexity,
    compute_spectral_features,
    generate_scaffold,
)


def test_generate_scaffold(sample_smiles: str) -> None:
    """Test scaffold generation from SMILES.

    Args:
        sample_smiles: Sample SMILES fixture.
    """
    scaffold = generate_scaffold(sample_smiles)
    assert isinstance(scaffold, str)
    assert len(scaffold) > 0


def test_compute_scaffold_complexity(sample_molecule: Chem.Mol) -> None:
    """Test scaffold complexity computation.

    Args:
        sample_molecule: Sample molecule fixture.
    """
    complexity = compute_scaffold_complexity(sample_molecule)
    assert isinstance(complexity, float)
    assert 0.0 <= complexity <= 1.0


def test_mol_to_graph(sample_molecule: Chem.Mol) -> None:
    """Test molecule to graph conversion.

    Args:
        sample_molecule: Sample molecule fixture.
    """
    data = mol_to_graph(sample_molecule, label=1.0)

    assert data.x.shape[0] == sample_molecule.GetNumAtoms()
    assert data.x.shape[1] == 6  # Number of atom features
    assert data.edge_index.shape[0] == 2
    assert data.y.item() == 1.0


def test_compute_graph_laplacian() -> None:
    """Test graph Laplacian computation."""
    # Simple triangle graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)

    num_nodes = 3

    # Test unnormalized Laplacian
    laplacian = compute_graph_laplacian(edge_index, num_nodes, normalize=False)
    assert laplacian.shape == (num_nodes, num_nodes)

    # Test normalized Laplacian
    laplacian_norm = compute_graph_laplacian(edge_index, num_nodes, normalize=True)
    assert laplacian_norm.shape == (num_nodes, num_nodes)


def test_compute_spectral_features() -> None:
    """Test spectral feature computation."""
    # Simple triangle graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)

    num_nodes = 3
    num_frequencies = 3

    eigenvalues, eigenvectors = compute_spectral_features(
        edge_index, num_nodes, num_frequencies
    )

    assert eigenvalues.shape == (num_frequencies,)
    assert eigenvectors.shape == (num_nodes, num_frequencies)


def test_invalid_smiles() -> None:
    """Test handling of invalid SMILES."""
    invalid_smiles = "invalid_smiles_123"
    scaffold = generate_scaffold(invalid_smiles)
    assert scaffold == ""
