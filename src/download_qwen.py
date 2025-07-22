#!/usr/bin/env python3
"""
Download Qwen3-0.6B model for testing.
"""

import logging
import sys
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_qwen_model():
    """Download Qwen model with progress tracking."""
    
    try:
        from huggingface_hub import snapshot_download
        
        logger.info("🔄 Starting Qwen3-0.6B download...")
        start_time = time.time()
        
        # Download to our models directory
        models_dir = Path(__file__).parent.parent / "models" / "qwen-0.6b"
        
        cache_path = snapshot_download(
            repo_id="Qwen/Qwen3-0.6B",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False  # Download actual files
        )
        
        end_time = time.time()
        download_time = end_time - start_time
        
        logger.info(f"✅ Download completed in {download_time:.1f} seconds")
        logger.info(f"📁 Model saved to: {cache_path}")
        
        # Check model files
        config_file = models_dir / "config.json"
        if config_file.exists():
            logger.info("✅ config.json found")
        
        model_files = list(models_dir.glob("*.safetensors")) + list(models_dir.glob("*.bin"))
        logger.info(f"✅ Found {len(model_files)} model weight files")
        
        return str(models_dir)
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return None

def main():
    """Main download function."""
    model_path = download_qwen_model()
    
    if model_path:
        logger.info("✅ Model download successful!")
        logger.info(f"📁 Path: {model_path}")
        return 0
    else:
        logger.error("❌ Model download failed!")
        return 1

if __name__ == "__main__":
    exit(main())