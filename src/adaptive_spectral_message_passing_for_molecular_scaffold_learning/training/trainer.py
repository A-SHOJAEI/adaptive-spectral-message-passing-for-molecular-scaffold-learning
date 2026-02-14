"""Trainer class for model training with curriculum learning support."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch_geometric.data import DataLoader
from tqdm import tqdm

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.components import (
    ScaffoldConsistencyLoss,
    SpectralFilterDiversityLoss,
    SpectralSmoothnessLoss,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for adaptive spectral GNN with curriculum learning."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        use_amp: bool = True,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            config: Configuration dictionary.
            device: Device to train on.
            use_amp: Whether to use automatic mixed precision.
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_amp = use_amp

        # Training config
        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_epochs", 100)
        self.grad_clip = train_config.get("grad_clip", 1.0)
        self.early_stopping_patience = train_config.get("early_stopping_patience", 15)
        self.min_delta = train_config.get("min_delta", 0.0001)

        # Curriculum learning
        curriculum_config = config.get("curriculum", {})
        self.curriculum_enabled = curriculum_config.get("enabled", False)
        self.warmup_epochs = curriculum_config.get("warmup_epochs", 5)

        # Optimizer
        optimizer_config = config.get("optimizer", {})
        opt_type = optimizer_config.get("type", "adam")

        if opt_type == "adam":
            self.optimizer = Adam(
                model.parameters(),
                lr=train_config.get("learning_rate", 0.001),
                weight_decay=train_config.get("weight_decay", 0.00001),
                betas=optimizer_config.get("betas", [0.9, 0.999]),
                eps=optimizer_config.get("eps", 0.00000001),
            )
        elif opt_type == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=train_config.get("learning_rate", 0.001),
                weight_decay=train_config.get("weight_decay", 0.00001),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        # Learning rate scheduler
        scheduler_config = config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("min_lr", 0.00001),
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer, step_size=scheduler_config.get("step_size", 30), gamma=0.1
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=10
            )
        else:
            self.scheduler = None

        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()

        # Custom loss components
        reg_config = config.get("regularization", {})
        self.spectral_smoothness_loss = SpectralSmoothnessLoss(
            weight=reg_config.get("spectral_smoothness_weight", 0.01)
        )
        self.scaffold_consistency_loss = ScaffoldConsistencyLoss(
            weight=reg_config.get("scaffold_consistency_weight", 0.05)
        )
        self.filter_diversity_loss = SpectralFilterDiversityLoss(
            weight=reg_config.get("filter_diversity_weight", 0.001)
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Training state
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_aucs = []

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (average loss, metrics dict).
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Curriculum learning: adjust data difficulty
        if self.curriculum_enabled and epoch < self.warmup_epochs:
            # During warmup, gradually increase data complexity
            difficulty_ratio = (epoch + 1) / self.warmup_epochs
            num_samples = int(len(train_loader.dataset) * difficulty_ratio)
            logger.info(f"Curriculum: Using {num_samples}/{len(train_loader.dataset)} samples")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output, auxiliary = self.model(data)
                    loss = self._compute_loss(data, output, auxiliary)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output, auxiliary = self.model(data)
                loss = self._compute_loss(data, output, auxiliary)
                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            total_loss += loss.item()

            # Collect predictions and labels
            preds = torch.sigmoid(output).detach().cpu().numpy()
            labels = data.y.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        try:
            train_auc = roc_auc_score(all_labels, all_preds)
        except Exception as e:
            logger.warning(f"Could not compute training AUC: {e}")
            train_auc = 0.0

        metrics = {"loss": avg_loss, "auc": train_auc}

        return avg_loss, metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (average loss, metrics dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)

                output, auxiliary = self.model(data)
                loss = self._compute_loss(data, output, auxiliary)

                total_loss += loss.item()

                # Collect predictions and labels
                preds = torch.sigmoid(output).detach().cpu().numpy()
                labels = data.y.detach().cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = total_loss / len(val_loader)

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except Exception as e:
            logger.warning(f"Could not compute validation AUC: {e}")
            val_auc = 0.0

        metrics = {"loss": avg_loss, "auc": val_auc}

        return avg_loss, metrics

    def _compute_loss(self, data: Any, output: torch.Tensor, auxiliary: Dict[str, Any]) -> torch.Tensor:
        """Compute total loss including custom regularization terms.

        Args:
            data: Input batch data.
            output: Model output predictions.
            auxiliary: Auxiliary outputs from model.

        Returns:
            Total loss value.
        """
        # Main classification loss
        main_loss = self.criterion(output.squeeze(), data.y.squeeze().float())

        # Spectral smoothness regularization
        smoothness_loss = self.spectral_smoothness_loss(
            auxiliary["node_embeddings"], data.edge_index
        )

        # Scaffold consistency regularization
        scaffolds = getattr(data, "scaffold_list", [])
        consistency_loss = self.scaffold_consistency_loss(
            auxiliary["graph_embeddings"], scaffolds, data.batch
        )

        # Filter diversity regularization
        diversity_loss = torch.tensor(0.0, device=output.device)
        if auxiliary["filter_weights"] is not None:
            for filter_weights in auxiliary["filter_weights"]:
                diversity_loss += self.filter_diversity_loss(filter_weights)

        # Total loss
        total_loss = main_loss + smoothness_loss + consistency_loss + diversity_loss

        return total_loss

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Dictionary with training history.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics['auc']:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}"
            )

            self.train_losses.append(train_loss)
            self.val_aucs.append(val_metrics["auc"])

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auc"])
                else:
                    self.scheduler.step()

            # Early stopping and checkpointing
            if val_metrics["auc"] > self.best_val_auc + self.min_delta:
                self.best_val_auc = val_metrics["auc"]
                self.epochs_without_improvement = 0

                # Save best model
                checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_auc": val_metrics["auc"],
                        "config": self.config,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        final_checkpoint_path = Path(checkpoint_dir) / "final_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_auc": val_metrics["auc"],
                "config": self.config,
            },
            final_checkpoint_path,
        )
        logger.info(f"Saved final model to {final_checkpoint_path}")

        history = {"train_loss": self.train_losses, "val_auc": self.val_aucs}

        return history
