#!/usr/bin/env python3
"""
Alternative startup script for Tea Leaf Detection Website
This script sets up the environment to handle PyTorch compatibility issues.
"""

import os
import sys

# Tell PyTorch to allow legacy checkpoint loading when Ultralytics needs it.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Import and run the main application
if __name__ == "__main__":
    try:
        # Import after setting environment
        import torch
        from ultralytics.nn.modules import Bottleneck, C2f, Conv, Detect, SPPF
        from ultralytics.nn.tasks import DetectionModel
        
        # Additional compatibility setup
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals([
                    DetectionModel,
                    Conv,
                    Bottleneck,
                    C2f,
                    SPPF,
                    Detect,
                ])
                print("✅ PyTorch safe globals configured for Ultralytics")
            except Exception as e:
                print(f"⚠️  Could not configure safe globals: {e}")
        
        # Import and run the main application
        from run import main
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)