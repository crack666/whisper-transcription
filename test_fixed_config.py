#!/usr/bin/env python3

import json
import logging
from pathlib import Path

# Simple test without imports to verify the config
def test_config():
    """Test the fixed configuration"""
    
    # Test lecture_fixed.json config
    config_path = Path("configs/lecture_fixed.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Testing lecture_fixed.json config:")
        print(f"   min_silence_len: {config.get('min_silence_len', 'not set')}ms")
        print(f"   silence_adjustment: {config.get('silence_adjustment', 'not set')}")
        print(f"   min_segment_length: {config.get('min_segment_length', 'not set')}ms")
        print(f"   overlap_duration: {config.get('overlap_duration', 'not set')}ms")
        
        # Validate critical parameters
        critical_params = {
            'min_silence_len': 2000,
            'silence_adjustment': 4.0,
            'min_segment_length': 2000
        }
        
        all_good = True
        for param, expected in critical_params.items():
            actual = config.get(param)
            if actual != expected:
                print(f"❌ ISSUE: {param} = {actual}, expected {expected}")
                all_good = False
        
        if all_good:
            print("✅ Configuration parameters look good!")
        else:
            print("❌ Configuration needs adjustment")
    else:
        print("❌ lecture_fixed.json not found")

if __name__ == "__main__":
    test_config()