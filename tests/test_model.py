"""Tests for model components and architecture."""

import pytest
import torch
from torch_geometric.data import Batch

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.components import (
    AdaptiveSpectralConv,
    ScaffoldConsistencyLoss,
    SpectralFilterDiversityLoss,
    SpectralSmoothnessLoss,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import (
    AdaptiveSpectralGNN,
)


def test_adaptive_spectral_conv() -> None:
    """Test adaptive spectral convolution layer."""
    in_channels = 16
    out_channels = 32
    num_frequencies = 8

    conv = AdaptiveSpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        num_frequencies=num_frequencies,
        adaptive=True,
    )

    # Create sample input
    x = torch.randn(10, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # Forward pass
    out, filter_weights = conv(x, edge_index)

    assert out.shape == (10, out_channels)
    assert filter_weights.shape == (num_frequencies,)


def test_adaptive_spectral_gnn_forward(sample_graph_data: Batch, sample_config: dict) -> None:
    """Test forward pass of AdaptiveSpectralGNN.

    Args:
        sample_graph_data: Sample graph data fixture.
        sample_config: Sample config fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)

    # Add batch dimension
    sample_graph_data.batch = torch.zeros(sample_graph_data.x.shape[0], dtype=torch.long)

    # Forward pass
    output, auxiliary = model(sample_graph_data)

    assert output.shape == (1, 1)  # (batch_size, output_dim)
    assert "node_embeddings" in auxiliary
    assert "graph_embeddings" in auxiliary
    assert "filter_weights" in auxiliary


def test_model_from_config(sample_config: dict) -> None:
    """Test model creation from config.

    Args:
        sample_config: Sample config fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)

    assert model.hidden_dim == sample_config["model"]["hidden_dim"]
    assert model.num_layers == sample_config["model"]["num_layers"]
    assert model.num_frequencies == sample_config["spectral"]["num_frequencies"]


def test_spectral_smoothness_loss() -> None:
    """Test spectral smoothness loss computation."""
    loss_fn = SpectralSmoothnessLoss(weight=0.01)

    node_embeddings = torch.randn(10, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    loss = loss_fn(node_embeddings, edge_index)

    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0.0


def test_scaffold_consistency_loss() -> None:
    """Test scaffold consistency loss computation."""
    loss_fn = ScaffoldConsistencyLoss(weight=0.05)

    embeddings = torch.randn(4, 32)
    scaffolds = ["scaffold1", "scaffold1", "scaffold2", "scaffold2"]
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = loss_fn(embeddings, scaffolds, batch)

    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0.0


def test_filter_diversity_loss() -> None:
    """Test spectral filter diversity loss computation."""
    loss_fn = SpectralFilterDiversityLoss(weight=0.001)

    filter_weights = torch.rand(8)

    loss = loss_fn(filter_weights)

    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0.0


def test_model_output_shape(sample_config: dict) -> None:
    """Test model output shapes with different batch sizes.

    Args:
        sample_config: Sample config fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)

    # Create batched input
    x = torch.randn(20, 6)  # 20 nodes, 6 features
    edge_index = torch.randint(0, 20, (2, 40))
    batch = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)  # 2 graphs

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=edge_index, batch=batch)

    output, _ = model(data)

    assert output.shape == (2, 1)  # 2 graphs in batch
