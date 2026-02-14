#!/usr/bin/env python
"""Training script - delegates to CLI module."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_spectral_message_passing_for_molecular_scaffold_learning.cli.train import main

if __name__ == "__main__":
    main()
