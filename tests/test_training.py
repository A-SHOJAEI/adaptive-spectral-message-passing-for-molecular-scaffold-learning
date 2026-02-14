"""Tests for training components."""

import pytest
import torch
from torch_geometric.loader import DataLoader

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import (
    AdaptiveSpectralGNN,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.training.trainer import (
    Trainer,
)


def test_trainer_initialization(sample_config: dict, device: torch.device) -> None:
    """Test trainer initialization.

    Args:
        sample_config: Sample config fixture.
        device: Device fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)
    trainer = Trainer(model, sample_config, device, use_amp=False)

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.criterion is not None


def test_trainer_train_epoch(
    sample_config: dict, sample_graph_data: torch.Tensor, device: torch.device
) -> None:
    """Test single training epoch.

    Args:
        sample_config: Sample config fixture.
        sample_graph_data: Sample graph data fixture.
        device: Device fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)
    trainer = Trainer(model, sample_config, device, use_amp=False)

    # Create minimal dataloader
    sample_graph_data.batch = torch.zeros(sample_graph_data.x.shape[0], dtype=torch.long)

    data_list = [sample_graph_data] * 4
    loader = DataLoader(data_list, batch_size=2)

    # Train one epoch
    loss, metrics = trainer.train_epoch(loader, epoch=0)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert "loss" in metrics
    assert "auc" in metrics


def test_trainer_validate(
    sample_config: dict, sample_graph_data: torch.Tensor, device: torch.device
) -> None:
    """Test validation.

    Args:
        sample_config: Sample config fixture.
        sample_graph_data: Sample graph data fixture.
        device: Device fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)
    trainer = Trainer(model, sample_config, device, use_amp=False)

    # Create minimal dataloader
    sample_graph_data.batch = torch.zeros(sample_graph_data.x.shape[0], dtype=torch.long)

    data_list = [sample_graph_data] * 4
    loader = DataLoader(data_list, batch_size=2)

    # Validate
    loss, metrics = trainer.validate(loader)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert "loss" in metrics
    assert "auc" in metrics


def test_optimizer_step(sample_config: dict, device: torch.device) -> None:
    """Test optimizer step.

    Args:
        sample_config: Sample config fixture.
        device: Device fixture.
    """
    model = AdaptiveSpectralGNN.from_config(sample_config)
    trainer = Trainer(model, sample_config, device, use_amp=False)

    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Create dummy batch
    from torch_geometric.data import Data

    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    y = torch.tensor([1.0], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)

    # Forward and backward
    trainer.optimizer.zero_grad()
    output, auxiliary = model(data)
    loss = trainer._compute_loss(data, output, auxiliary)
    loss.backward()
    trainer.optimizer.step()

    # Check that parameters changed
    updated_params = list(model.parameters())
    params_changed = any(
        not torch.equal(p1, p2) for p1, p2 in zip(initial_params, updated_params)
    )

    assert params_changed, "Parameters should change after optimizer step"


def test_gradient_clipping(sample_config: dict, device: torch.device) -> None:
    """Test gradient clipping.

    Args:
        sample_config: Sample config fixture.
        device: Device fixture.
    """
    sample_config["training"]["grad_clip"] = 0.5

    model = AdaptiveSpectralGNN.from_config(sample_config)
    trainer = Trainer(model, sample_config, device, use_amp=False)

    # Create dummy batch
    from torch_geometric.data import Data

    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    y = torch.tensor([1.0], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)

    # Forward and backward
    trainer.optimizer.zero_grad()
    output, auxiliary = model(data)
    loss = trainer._compute_loss(data, output, auxiliary)
    loss.backward()

    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)

    # Check that gradients are clipped
    total_norm = torch.sqrt(
        sum(p.grad.detach().norm() ** 2 for p in model.parameters() if p.grad is not None)
    )

    assert total_norm <= trainer.grad_clip + 1e-5  # Small tolerance for numerical errors
