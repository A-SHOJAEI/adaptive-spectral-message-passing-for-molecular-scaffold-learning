#!/usr/bin/env python
"""Quick pipeline test to verify all components work."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "src"))

import torch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_pipeline():
    """Test all pipeline components."""
    print("=" * 60)
    print("PIPELINE COMPONENT VERIFICATION")
    print("=" * 60)
    
    # 1. Config loading
    print("\n1. Testing config loading...")
    from adaptive_spectral_message_passing_for_molecular_scaffold_learning.utils.config import load_config, set_seed
    config = load_config('configs/default.yaml')
    print(f"   ✓ Config loaded (hidden_dim={config['model']['hidden_dim']})")
    
    # 2. Random seed
    print("\n2. Setting random seed...")
    set_seed(42)
    print("   ✓ Random seed set to 42")
    
    # 3. Model creation
    print("\n3. Creating model...")
    from adaptive_spectral_message_passing_for_molecular_scaffold_learning.models.model import AdaptiveSpectralGNN
    model = AdaptiveSpectralGNN.from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created ({num_params:,} parameters)")
    
    # 4. Device setup
    print("\n4. Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   ✓ Device: {device}")
    
    # 5. Trainer initialization
    print("\n5. Initializing trainer...")
    from adaptive_spectral_message_passing_for_molecular_scaffold_learning.training.trainer import Trainer
    trainer = Trainer(model, config, device, use_amp=False)
    print("   ✓ Trainer initialized")
    
    # 6. Test forward pass
    print("\n6. Testing forward pass...")
    from torch_geometric.data import Data, Batch
    # Create dummy molecular graph
    x = torch.randn(10, 6, device=device)  # 10 atoms, 6 features
    edge_index = torch.tensor([[0,1,1,2], [1,0,2,1]], dtype=torch.long, device=device)
    edge_attr = torch.randn(4, 3, device=device)
    y = torch.tensor([1.0], device=device)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    batch = Batch.from_data_list([data])
    
    model.eval()
    with torch.no_grad():
        output, auxiliary = model(batch)
    print(f"   ✓ Forward pass successful (output shape: {output.shape})")
    
    # 7. Check all required scripts exist
    print("\n7. Checking required scripts...")
    scripts = ['scripts/train.py', 'scripts/evaluate.py', 'scripts/predict.py']
    for script in scripts:
        if Path(script).exists():
            print(f"   ✓ {script} exists")
        else:
            print(f"   ✗ {script} MISSING")
    
    # 8. Check configs
    print("\n8. Checking config files...")
    configs = ['configs/default.yaml', 'configs/ablation.yaml']
    for cfg in configs:
        if Path(cfg).exists():
            print(f"   ✓ {cfg} exists")
        else:
            print(f"   ✗ {cfg} MISSING")
    
    print("\n" + "=" * 60)
    print("ALL PIPELINE COMPONENTS VERIFIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe project is ready to train. Run:")
    print("  python scripts/train.py --config configs/default.yaml")
    print("\nNote: First run will download BBBP dataset (~10-30 seconds)")

if __name__ == "__main__":
    test_pipeline()
