#!/bin/bash

echo "🔧 Testing Enhanced Transcriber System"
echo "======================================"

# Navigate to project directory
cd /mnt/c/Users/crack.crackdesk/source/repos/tools/whisper/whisper-transcription

echo "📍 Current directory: $(pwd)"
echo "📁 Files available:"
ls -la *.mp3 *.py 2>/dev/null || echo "   No audio/python files found"

echo ""
echo "🐍 Python Environment Check:"
echo "   Python: $(which python)"
echo "   Version: $(python --version)"

echo ""
echo "📦 Dependency Check:"
python -c "
try:
    import whisper
    print('   ✅ whisper:', whisper.__version__)
except ImportError:
    print('   ❌ whisper: Not installed')

try:
    import pydub
    print('   ✅ pydub: Available')
except ImportError:
    print('   ❌ pydub: Not installed')

try:
    import torch
    print('   ✅ torch:', torch.__version__)
except ImportError:
    print('   ❌ torch: Not installed')
"

echo ""
echo "🎤 Testing Enhanced Transcriber:"

# Test 1: Simple configuration test
echo "Test 1: Configuration validation"
python test_fixed_config.py

echo ""
echo "Test 2: Enhanced Transcriber with interview.mp3"
python test_enhanced_final.py

echo ""
echo "🎯 Alternative: Quick transcription test"
echo "   You can also try:"
echo "   python quick_transcribe.py --input interview.mp3 --output test_result.json --verbose"