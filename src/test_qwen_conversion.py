#!/usr/bin/env python3
"""
Quick test to check if anemll conversion pipeline works with a sample model.
This tests the conversion infrastructure before running full baseline tests.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_anemll_conversion():
    """Test anemll conversion infrastructure."""
    
    # Check if anemll is properly installed
    anemll_path = Path(__file__).parent.parent / "anemll"
    if not anemll_path.exists():
        logger.error("❌ anemll directory not found")
        return False
    
    conversion_script = anemll_path / "anemll" / "utils" / "convert_model.sh"
    if not conversion_script.exists():
        logger.error(f"❌ Conversion script not found: {conversion_script}")
        return False
    
    logger.info(f"✅ Found anemll conversion script: {conversion_script}")
    
    # Test script accessibility
    if not os.access(conversion_script, os.X_OK):
        logger.info("📝 Making conversion script executable...")
        os.chmod(conversion_script, 0o755)
    
    # Test help output
    try:
        import subprocess
        result = subprocess.run(
            [str(conversion_script), "--help"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        if "Usage:" in result.stdout or result.returncode == 1:  # Help usually returns 1
            logger.info("✅ Conversion script is accessible and shows help")
            return True
        else:
            logger.error(f"❌ Unexpected script output: {result.stdout[:200]}...")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Conversion script timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing conversion script: {e}")
        return False

def test_dependencies():
    """Test required dependencies for anemll."""
    
    required_packages = [
        'torch',
        'transformers', 
        'coremltools'
    ]
    
    missing = []
    versions = {}
    
    for package in required_packages:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                versions[package] = module.__version__
            else:
                versions[package] = "unknown"
            logger.info(f"✅ {package}: {versions[package]}")
        except ImportError:
            missing.append(package)
            logger.error(f"❌ Missing package: {package}")
    
    if missing:
        logger.error(f"❌ Missing required packages: {missing}")
        logger.info("💡 Install with: pip3 install torch transformers coremltools")
        return False
    
    # Check CoreML Tools version
    try:
        import coremltools
        version = coremltools.__version__
        major_version = int(version.split('.')[0])
        if major_version < 8:
            logger.warning(f"⚠️ CoreML Tools version {version} may be too old (recommended: 8.2+)")
        else:
            logger.info(f"✅ CoreML Tools version {version} looks good")
    except Exception as e:
        logger.error(f"❌ Error checking CoreML Tools: {e}")
    
    return True

def test_model_config_download():
    """Test downloading just the model config to verify HuggingFace access."""
    
    try:
        from transformers import AutoConfig
        logger.info("🔄 Testing HuggingFace model access (config only)...")
        
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        logger.info(f"✅ Successfully loaded Qwen config: {config.model_type}")
        logger.info(f"📊 Model specs: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load Qwen config: {e}")
        logger.info("💡 This might indicate network issues or HuggingFace access problems")
        return False

def main():
    """Run all conversion infrastructure tests."""
    
    logger.info("🎯 Testing anemll conversion infrastructure...")
    logger.info("="*60)
    
    tests = [
        ("Dependencies Check", test_dependencies),
        ("anemll Conversion Script", test_anemll_conversion),
        ("HuggingFace Access", test_model_config_download),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📝 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎯 ANEMLL CONVERSION TEST RESULTS")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests passed! Ready for model conversion.")
        return 0
    else:
        logger.info("⚠️ Some tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())