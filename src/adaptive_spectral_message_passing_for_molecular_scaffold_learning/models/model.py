"""Main model implementation for adaptive spectral GNN."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.components import (
    AdaptiveSpectralConv,
)

logger = logging.getLogger(__name__)


class AdaptiveSpectralGNN(nn.Module):
    """Adaptive Spectral Graph Neural Network for molecular property prediction.

    This model combines:
    1. Adaptive spectral graph convolutions that learn task-specific frequency filters
    2. Multi-layer message passing with residual connections
    3. Flexible graph-level readout (mean, max, or add pooling)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        num_spectral_filters: int = 8,
        num_frequencies: int = 16,
        dropout: float = 0.2,
        spectral_dropout: float = 0.1,
        use_batch_norm: bool = True,
        adaptive_filter: bool = True,
        readout: str = "mean",
    ):
        """Initialize Adaptive Spectral GNN.

        Args:
            input_dim: Dimension of input node features.
            hidden_dim: Dimension of hidden layers.
            output_dim: Dimension of output (1 for binary classification).
            num_layers: Number of graph convolution layers.
            num_spectral_filters: Number of spectral filters per layer.
            num_frequencies: Number of spectral frequencies to use.
            dropout: Dropout rate for regular layers.
            spectral_dropout: Dropout rate for spectral coefficients.
            use_batch_norm: Whether to use batch normalization.
            adaptive_filter: Whether to use adaptive frequency filtering.
            readout: Type of graph-level pooling ('mean', 'max', 'add').
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_frequencies = num_frequencies
        self.adaptive_filter = adaptive_filter
        self.readout = readout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Spectral convolution layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for _ in range(num_layers):
            conv = AdaptiveSpectralConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_frequencies=num_frequencies,
                adaptive=adaptive_filter,
                dropout=spectral_dropout,
            )
            self.conv_layers.append(conv)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Readout pooling
        if readout == "mean":
            self.pool = global_mean_pool
        elif readout == "max":
            self.pool = global_max_pool
        elif readout == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown readout type: {readout}")

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable parameters."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for conv in self.conv_layers:
            conv.reset_parameters()

        if self.batch_norms is not None:
            for bn in self.batch_norms:
                bn.reset_parameters()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, data: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass of the model.

        Args:
            data: PyTorch Geometric Batch object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes]
                - spectral_eigenvalues: Optional [num_graphs, num_frequencies]
                - spectral_eigenvectors: Optional [num_nodes, num_frequencies]

        Returns:
            Tuple of (predictions, auxiliary_outputs) where:
                - predictions: Output predictions [batch_size, output_dim]
                - auxiliary_outputs: Dict with intermediate values for analysis
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Get spectral features if available (as lists from custom collate)
        spectral_eigenvalues_list = getattr(data, "spectral_eigenvalues_list", None)
        spectral_eigenvectors_list = getattr(data, "spectral_eigenvectors_list", None)

        # For now, set to None (model will handle missing spectral features)
        # TODO: Properly handle per-graph spectral features in batches
        spectral_eigenvalues = None
        spectral_eigenvectors = None

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Store filter weights for each layer
        all_filter_weights = []

        # Apply spectral convolution layers with residual connections
        for i, conv in enumerate(self.conv_layers):
            # Store input for residual
            identity = x

            # Apply spectral convolution
            x, filter_weights = conv(
                x, edge_index, spectral_eigenvalues, spectral_eigenvectors
            )

            # Batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

            # Activation and dropout
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection
            x = x + identity

            # Store filter weights
            all_filter_weights.append(filter_weights)

        # Graph-level pooling
        graph_embeddings = self.pool(x, batch)

        # Output prediction
        out = self.fc1(graph_embeddings)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        # Prepare auxiliary outputs
        auxiliary = {
            "node_embeddings": x,
            "graph_embeddings": graph_embeddings,
            "filter_weights": torch.stack(all_filter_weights) if all_filter_weights else None,
        }

        return out, auxiliary

    def get_graph_embeddings(self, data: Batch) -> Tensor:
        """Extract graph-level embeddings without final classification layer.

        Args:
            data: PyTorch Geometric Batch object.

        Returns:
            Graph embeddings of shape [batch_size, hidden_dim].
        """
        _, auxiliary = self.forward(data)
        return auxiliary["graph_embeddings"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AdaptiveSpectralGNN":
        """Create model from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Initialized model instance.
        """
        model_config = config.get("model", {})
        spectral_config = config.get("spectral", {})

        return cls(
            input_dim=6,  # Fixed for molecular graphs (atom features)
            hidden_dim=model_config.get("hidden_dim", 128),
            output_dim=1,  # Binary classification
            num_layers=model_config.get("num_layers", 4),
            num_spectral_filters=model_config.get("num_spectral_filters", 8),
            num_frequencies=spectral_config.get("num_frequencies", 16),
            dropout=model_config.get("dropout", 0.2),
            spectral_dropout=model_config.get("spectral_dropout", 0.1),
            use_batch_norm=model_config.get("use_batch_norm", True),
            adaptive_filter=spectral_config.get("adaptive_filter", True),
            readout=model_config.get("readout", "mean"),
        )
