"""Data loading utilities for MoleculeNet datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import deepchem as dc
import numpy as np
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data, DataLoader
from tqdm import tqdm

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.data.preprocessing import (
    add_spectral_features_to_data,
    compute_scaffold_complexity,
    generate_scaffold,
)

logger = logging.getLogger(__name__)


def mol_to_graph(mol: Chem.Mol, label: Optional[float] = None) -> Data:
    """Convert RDKit molecule to PyTorch Geometric Data object.

    Args:
        mol: RDKit molecule object.
        label: Optional label for the molecule.

    Returns:
        PyTorch Geometric Data object.
    """
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices and features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())

        # Add both directions for undirected graph
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append([bond_type, is_conjugated, is_in_ring])
        edge_features.append([bond_type, is_conjugated, is_in_ring])

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def load_moleculenet_dataset(
    dataset_name: str = "BBBP", data_dir: str = "data"
) -> Tuple[List[Data], List[str]]:
    """Load MoleculeNet dataset and convert to PyG format.

    Args:
        dataset_name: Name of MoleculeNet dataset (e.g., 'BBBP', 'Tox21', 'BACE').
        data_dir: Directory to store/load data.

    Returns:
        Tuple of (graph_list, smiles_list) where graph_list contains PyG Data objects.
    """
    import pandas as pd
    import os

    logger.info(f"Loading {dataset_name} dataset from MoleculeNet...")

    # Download dataset if needed using DeepChem, then load directly from CSV
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    csv_file = data_path / "BBBP.csv"

    # Download if not exists
    if not csv_file.exists():
        logger.info("Downloading BBBP dataset...")
        import urllib.request
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        urllib.request.urlretrieve(url, csv_file)

    # Load CSV directly
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} molecules from CSV")

    # Extract molecules and labels
    smiles_list = []
    graph_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting molecules to graphs"):
        try:
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                continue

            label = float(row['p_np'])  # Binary classification label

            # Convert to graph
            data = mol_to_graph(mol, label)

            # Add scaffold information
            scaffold = generate_scaffold(smiles)
            complexity = compute_scaffold_complexity(mol)

            data.smiles = smiles
            data.scaffold = scaffold
            data.complexity = torch.tensor([complexity], dtype=torch.float)

            graph_list.append(data)
            smiles_list.append(smiles)

        except Exception as e:
            logger.warning(f"Error processing molecule {idx}: {e}")
            continue

    logger.info(f"Successfully loaded {len(graph_list)} molecules")
    return graph_list, smiles_list


def scaffold_split(
    graphs: List[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Split dataset by scaffold to test generalization to new scaffolds.

    Args:
        graphs: List of PyG Data objects.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed.

    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs).
    """
    # Group molecules by scaffold
    scaffold_to_graphs: Dict[str, List[Data]] = {}
    for graph in graphs:
        scaffold = graph.scaffold
        if scaffold not in scaffold_to_graphs:
            scaffold_to_graphs[scaffold] = []
        scaffold_to_graphs[scaffold].append(graph)

    # Sort scaffolds by size (number of molecules)
    scaffolds = sorted(scaffold_to_graphs.keys(), key=lambda s: len(scaffold_to_graphs[s]))

    # Split scaffolds into train/val/test
    train_graphs = []
    val_graphs = []
    test_graphs = []

    train_size = int(len(graphs) * train_ratio)
    val_size = int(len(graphs) * val_ratio)

    np.random.seed(seed)
    np.random.shuffle(scaffolds)

    for scaffold in scaffolds:
        graphs_in_scaffold = scaffold_to_graphs[scaffold]

        if len(train_graphs) < train_size:
            train_graphs.extend(graphs_in_scaffold)
        elif len(val_graphs) < val_size:
            val_graphs.extend(graphs_in_scaffold)
        else:
            test_graphs.extend(graphs_in_scaffold)

    logger.info(
        f"Scaffold split: {len(train_graphs)} train, "
        f"{len(val_graphs)} val, {len(test_graphs)} test"
    )

    return train_graphs, val_graphs, test_graphs


def sort_by_complexity(graphs: List[Data]) -> List[Data]:
    """Sort molecules by scaffold complexity for curriculum learning.

    Args:
        graphs: List of PyG Data objects.

    Returns:
        Sorted list of graphs from simple to complex.
    """
    return sorted(graphs, key=lambda g: g.complexity.item())


def custom_collate(data_list: List[Data]) -> Batch:
    """Custom collate function that preserves scaffold information.

    Args:
        data_list: List of PyG Data objects.

    Returns:
        Batched data with preserved scaffold information.
    """
    # Extract scaffolds and other string attributes before batching
    scaffolds = []
    smiles_list = []

    for data in data_list:
        scaffolds.append(data.scaffold if hasattr(data, "scaffold") else "")
        smiles_list.append(data.smiles if hasattr(data, "smiles") else "")

    # Create copies and remove string attributes to avoid batching issues
    cleaned_data_list = []
    for data in data_list:
        # Clone the data object
        data_clone = data.clone()
        # Remove string attributes that can't be batched
        if hasattr(data_clone, "scaffold"):
            delattr(data_clone, "scaffold")
        if hasattr(data_clone, "smiles"):
            delattr(data_clone, "smiles")
        cleaned_data_list.append(data_clone)

    # Batch the cleaned data
    batch = Batch.from_data_list(cleaned_data_list)

    # Add string attributes back as lists
    batch.scaffold_list = scaffolds
    batch.smiles_list = smiles_list

    return batch


def get_data_loaders(
    config: Dict[str, Any], add_spectral: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders from config.

    Args:
        config: Configuration dictionary.
        add_spectral: Whether to precompute spectral features.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset", "BBBP")
    split_type = data_config.get("split_type", "scaffold")
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = data_config.get("num_workers", 0)  # Use 0 to avoid multiprocessing issues

    # Load dataset
    graphs, smiles = load_moleculenet_dataset(dataset_name)

    # Note: Spectral features will be computed on-the-fly during forward pass
    # Precomputing them causes batching issues due to variable graph sizes
    if add_spectral:
        logger.info("Spectral features will be computed on-the-fly (skipping precomputation)")

    # Split dataset
    train_graphs, val_graphs, test_graphs = scaffold_split(
        graphs,
        train_ratio=data_config.get("train_ratio", 0.8),
        val_ratio=data_config.get("val_ratio", 0.1),
        test_ratio=data_config.get("test_ratio", 0.1),
        seed=config.get("seed", 42),
    )

    # Apply curriculum sorting if enabled
    curriculum_config = config.get("curriculum", {})
    if curriculum_config.get("enabled", False):
        logger.info("Sorting training data by complexity for curriculum learning")
        train_graphs = sort_by_complexity(train_graphs)

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=not curriculum_config.get("enabled", False),  # Don't shuffle if curriculum
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=custom_collate,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=custom_collate,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=custom_collate,
    )

    logger.info(
        f"Created data loaders: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches, {len(test_loader)} test batches"
    )

    return train_loader, val_loader, test_loader
