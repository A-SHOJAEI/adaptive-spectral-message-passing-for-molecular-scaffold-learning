"""Data preprocessing utilities for molecular graphs."""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def generate_scaffold(smiles: str) -> str:
    """Generate Murcko scaffold from SMILES string.

    Args:
        smiles: SMILES representation of molecule.

    Returns:
        SMILES string of the scaffold.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception as e:
        logger.warning(f"Failed to generate scaffold for {smiles}: {e}")
        return ""


def compute_scaffold_complexity(mol: Chem.Mol) -> float:
    """Compute scaffold complexity metric based on structural features.

    Complexity is measured by:
    - Number of rings
    - Number of aromatic rings
    - Molecular weight
    - Number of rotatable bonds

    Args:
        mol: RDKit molecule object.

    Returns:
        Normalized complexity score between 0 and 1.
    """
    try:
        from rdkit.Chem import Descriptors, Lipinski

        num_rings = Lipinski.NumRings(mol)
        num_aromatic_rings = Lipinski.NumAromaticRings(mol)
        mol_weight = Descriptors.MolWt(mol)
        num_rotatable = Lipinski.NumRotatableBonds(mol)

        # Normalize components
        ring_score = min(num_rings / 6.0, 1.0)
        aromatic_score = min(num_aromatic_rings / 4.0, 1.0)
        weight_score = min(mol_weight / 500.0, 1.0)
        rotatable_score = min(num_rotatable / 10.0, 1.0)

        # Weighted combination
        complexity = (
            0.4 * ring_score + 0.3 * aromatic_score + 0.2 * weight_score + 0.1 * rotatable_score
        )

        return complexity
    except Exception as e:
        logger.warning(f"Failed to compute scaffold complexity: {e}")
        return 0.5


def compute_graph_laplacian(
    edge_index: torch.Tensor, num_nodes: int, normalize: bool = True
) -> torch.Tensor:
    """Compute graph Laplacian matrix.

    Args:
        edge_index: Edge indices of shape [2, num_edges].
        num_nodes: Number of nodes in the graph.
        normalize: Whether to compute normalized Laplacian.

    Returns:
        Laplacian matrix of shape [num_nodes, num_nodes].
    """
    # Build adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0

    # Compute degree matrix
    degree = adj.sum(dim=1)

    if normalize:
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        degree_mat = torch.diag(degree_inv_sqrt)
        laplacian = torch.eye(num_nodes) - degree_mat @ adj @ degree_mat
    else:
        # Unnormalized Laplacian: L = D - A
        laplacian = torch.diag(degree) - adj

    return laplacian


def compute_spectral_features(
    edge_index: torch.Tensor, num_nodes: int, num_frequencies: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spectral features using graph Laplacian eigendecomposition.

    Args:
        edge_index: Edge indices of shape [2, num_edges].
        num_nodes: Number of nodes in the graph.
        num_frequencies: Number of top eigenvectors to compute.

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
            - eigenvalues: Tensor of shape [num_frequencies]
            - eigenvectors: Tensor of shape [num_nodes, num_frequencies]
    """
    laplacian = compute_graph_laplacian(edge_index, num_nodes, normalize=True)

    # Compute eigendecomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        # Take first num_frequencies eigenpairs (smallest eigenvalues)
        k = min(num_frequencies, num_nodes)
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

        return eigenvalues, eigenvectors
    except Exception as e:
        logger.warning(f"Eigendecomposition failed: {e}. Using identity features.")
        eigenvalues = torch.arange(num_frequencies, dtype=torch.float32)
        eigenvectors = torch.eye(num_nodes, num_frequencies)
        return eigenvalues, eigenvectors


def add_spectral_features_to_data(data: Data, num_frequencies: int = 16) -> Data:
    """Add precomputed spectral features to PyG Data object.

    Args:
        data: PyTorch Geometric Data object.
        num_frequencies: Number of spectral frequencies to compute.

    Returns:
        Data object with added spectral_eigenvalues and spectral_eigenvectors.
    """
    eigenvalues, eigenvectors = compute_spectral_features(
        data.edge_index, data.num_nodes, num_frequencies
    )

    data.spectral_eigenvalues = eigenvalues
    data.spectral_eigenvectors = eigenvectors

    return data


def compute_functional_groups(mol: Chem.Mol) -> Dict[str, int]:
    """Compute functional group counts for a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Dictionary mapping functional group names to counts.
    """
    from rdkit.Chem import Fragments

    functional_groups = {
        "alkyl_halide": Fragments.fr_halogen(mol),
        "alcohol": Fragments.fr_Al_OH(mol),
        "aldehyde": Fragments.fr_aldehyde(mol),
        "ketone": Fragments.fr_ketone(mol),
        "carboxylic_acid": Fragments.fr_COO(mol),
        "ester": Fragments.fr_ester(mol),
        "ether": Fragments.fr_ether(mol),
        "amine": Fragments.fr_NH2(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH0(mol),
        "amide": Fragments.fr_amide(mol),
        "nitro": Fragments.fr_nitro(mol),
        "aromatic": Fragments.fr_benzene(mol),
    }

    return functional_groups
