"""Adaptive Spectral Message Passing for Molecular Scaffold Learning.

This package implements molecular property prediction using adaptive spectral
graph convolutions with scaffold-aware curriculum learning.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import (
    AdaptiveSpectralGNN,
)
from adaptive_spectral_message_passing_for_molecular_scaffold_learning.training.trainer import (
    Trainer,
)

__all__ = ["AdaptiveSpectralGNN", "Trainer"]
