#!/usr/bin/env python3
"""
Direct Syntax Check and Import Test
"""

import sys
import os
import traceback

print("🔧 DIRECT SYNTAX AND IMPORT TEST")
print("=" * 40)

# Check Python environment
print(f"📍 Python: {sys.version}")
print(f"📍 Working dir: {os.getcwd()}")

# Add src to path
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print(f"📍 Added to path: {src_path}")

# Test 1: Check if waveform_analyzer.py file exists
waveform_file = os.path.join('src', 'waveform_analyzer.py')
print(f"\n📁 Checking file: {waveform_file}")
if os.path.exists(waveform_file):
    print("✅ File exists")
    
    # Read first few lines to check basic syntax
    try:
        with open(waveform_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
        print("✅ File readable")
        print("📖 First 10 lines:")
        for i, line in enumerate(first_lines, 1):
            if line:
                print(f"   {i}: {line[:60]}{'...' if len(line) > 60 else ''}")
    except Exception as e:
        print(f"❌ File read error: {e}")
else:
    print("❌ File not found")

# Test 2: Syntax check with compile
print(f"\n🔍 SYNTAX CHECK")
try:
    with open(waveform_file, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    compile(source_code, waveform_file, 'exec')
    print("✅ Syntax is valid")
except SyntaxError as e:
    print(f"❌ Syntax Error:")
    print(f"   File: {e.filename}")
    print(f"   Line: {e.lineno}")
    print(f"   Error: {e.msg}")
    print(f"   Text: {e.text}")
except Exception as e:
    print(f"❌ Other error: {e}")

# Test 3: Import test
print(f"\n📦 IMPORT TEST")
try:
    import waveform_analyzer
    print("✅ waveform_analyzer module imported")
    
    # Check what's in the module
    print("📋 Module contents:")
    for attr in dir(waveform_analyzer):
        if not attr.startswith('_'):
            print(f"   - {attr}")
    
    # Try to create WaveformAnalyzer
    analyzer = waveform_analyzer.WaveformAnalyzer()
    print("✅ WaveformAnalyzer instance created")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ Other Error: {e}")
    traceback.print_exc()

print(f"\n🏁 TEST COMPLETED")
