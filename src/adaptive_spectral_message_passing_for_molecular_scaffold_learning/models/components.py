"""Custom model components including layers and loss functions."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

logger = logging.getLogger(__name__)


class AdaptiveSpectralConv(MessagePassing):
    """Adaptive spectral graph convolution layer.

    This layer learns task-specific frequency filters that adaptively weight
    different graph frequencies based on the input features and learned parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frequencies: int = 16,
        adaptive: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize adaptive spectral convolution layer.

        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
            num_frequencies: Number of spectral frequencies to use.
            adaptive: Whether to use adaptive frequency filtering.
            dropout: Dropout rate for spectral coefficients.
        """
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frequencies = num_frequencies
        self.adaptive = adaptive

        # Learnable spectral filter weights
        if adaptive:
            self.filter_network = nn.Sequential(
                nn.Linear(in_channels, num_frequencies),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_frequencies, num_frequencies),
                nn.Sigmoid(),  # Weights in [0, 1]
            )
        else:
            # Fixed uniform weights
            self.register_buffer(
                "fixed_weights", torch.ones(num_frequencies) / num_frequencies
            )

        # Transform input features
        self.lin = nn.Linear(in_channels, out_channels)

        # Frequency-specific transformations
        self.frequency_transforms = nn.ModuleList(
            [nn.Linear(out_channels, out_channels) for _ in range(num_frequencies)]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        if self.adaptive:
            for layer in self.filter_network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

        for transform in self.frequency_transforms:
            nn.init.xavier_uniform_(transform.weight)
            nn.init.zeros_(transform.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        spectral_eigenvalues: Optional[Tensor] = None,
        spectral_eigenvectors: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of adaptive spectral convolution.

        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            spectral_eigenvalues: Graph Laplacian eigenvalues [num_frequencies].
            spectral_eigenvectors: Graph Laplacian eigenvectors [num_nodes, num_frequencies].

        Returns:
            Tuple of (output features, filter weights).
        """
        # Transform input features
        x = self.lin(x)

        # Compute adaptive filter weights
        if self.adaptive:
            # Use global mean pooling of node features to compute filter weights
            global_features = x.mean(dim=0, keepdim=True)  # [1, out_channels]
            filter_weights = self.filter_network(
                global_features[:, : self.in_channels]
            )  # [1, num_frequencies]
            filter_weights = filter_weights.squeeze(0)  # [num_frequencies]
        else:
            filter_weights = self.fixed_weights

        # Apply spectral filtering
        if spectral_eigenvectors is not None and spectral_eigenvalues is not None:
            # Project to spectral domain
            k = min(self.num_frequencies, spectral_eigenvectors.shape[1])
            eigenvecs = spectral_eigenvectors[:, :k]  # [num_nodes, k]

            # Apply frequency-specific transformations
            spectral_out = torch.zeros_like(x)
            for i in range(k):
                # Transform features in this frequency
                freq_features = self.frequency_transforms[i](x)

                # Weight by eigenvector and filter weight
                weighted = eigenvecs[:, i : i + 1] * freq_features * filter_weights[i]
                spectral_out = spectral_out + weighted

            x = spectral_out

        # Standard message passing
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)

        return out, filter_weights

    def message(self, x_j: Tensor) -> Tensor:
        """Construct messages from neighbors.

        Args:
            x_j: Features of neighboring nodes.

        Returns:
            Messages to be aggregated.
        """
        return x_j


class SpectralSmoothnessLoss(nn.Module):
    """Loss that encourages spectral smoothness in learned representations."""

    def __init__(self, weight: float = 0.01):
        """Initialize spectral smoothness loss.

        Args:
            weight: Weight for this loss term.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
        spectral_eigenvalues: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute spectral smoothness loss.

        Args:
            node_embeddings: Node embeddings of shape [num_nodes, hidden_dim].
            edge_index: Edge indices of shape [2, num_edges].
            spectral_eigenvalues: Optional eigenvalues for weighting.

        Returns:
            Scalar loss value.
        """
        # Compute difference between connected nodes
        source, target = edge_index
        diff = node_embeddings[source] - node_embeddings[target]
        smoothness_loss = torch.mean(torch.sum(diff**2, dim=1))

        return self.weight * smoothness_loss


class ScaffoldConsistencyLoss(nn.Module):
    """Loss that encourages consistent predictions within the same scaffold."""

    def __init__(self, weight: float = 0.05):
        """Initialize scaffold consistency loss.

        Args:
            weight: Weight for this loss term.
        """
        super().__init__()
        self.weight = weight

    def forward(self, embeddings: Tensor, scaffolds: list, batch: Tensor) -> Tensor:
        """Compute scaffold consistency loss.

        Args:
            embeddings: Graph-level embeddings of shape [batch_size, hidden_dim].
            scaffolds: List of scaffold identifiers for each graph.
            batch: Batch assignment vector.

        Returns:
            Scalar loss value.
        """
        if len(scaffolds) != embeddings.shape[0]:
            return torch.tensor(0.0, device=embeddings.device)

        # Group embeddings by scaffold
        scaffold_to_indices = {}
        for i, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = []
            scaffold_to_indices[scaffold].append(i)

        # Compute variance within each scaffold
        total_loss = 0.0
        num_scaffolds = 0

        for scaffold, indices in scaffold_to_indices.items():
            if len(indices) > 1:
                scaffold_embeddings = embeddings[indices]
                mean_embedding = scaffold_embeddings.mean(dim=0, keepdim=True)
                variance = torch.mean((scaffold_embeddings - mean_embedding) ** 2)
                total_loss += variance
                num_scaffolds += 1

        if num_scaffolds == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return self.weight * (total_loss / num_scaffolds)


class SpectralFilterDiversityLoss(nn.Module):
    """Loss that encourages diversity in learned spectral filters."""

    def __init__(self, weight: float = 0.001):
        """Initialize spectral filter diversity loss.

        Args:
            weight: Weight for this loss term.
        """
        super().__init__()
        self.weight = weight

    def forward(self, filter_weights: Tensor) -> Tensor:
        """Compute filter diversity loss.

        Encourages filter weights to be diverse rather than uniform.

        Args:
            filter_weights: Filter weights of shape [num_frequencies].

        Returns:
            Scalar loss value.
        """
        # Compute entropy of filter weights (higher entropy = more diverse)
        # Negative entropy as loss (minimize = maximize entropy)
        normalized_weights = F.softmax(filter_weights, dim=0)
        entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8))

        # We want high entropy, so minimize negative entropy
        max_entropy = torch.log(torch.tensor(len(filter_weights), dtype=torch.float))
        diversity_loss = max_entropy - entropy

        return self.weight * diversity_loss
