#!/usr/bin/env python3
"""
Quick test for precision waveform mode.
"""
import sys
import os
sys.path.append('.')

def main():
    print("🔬 Testing Precision Waveform Mode")
    print("=" * 50)
    
    # Test 1: Import WaveformAnalyzer
    try:
        from src.waveform_analyzer import WaveformAnalyzer, PRECISION_CONFIG
        print("✅ WaveformAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ WaveformAnalyzer import failed: {e}")
        return False
    
    # Test 2: Initialize with precision config
    try:
        analyzer = WaveformAnalyzer(PRECISION_CONFIG)
        print("✅ WaveformAnalyzer initialized with PRECISION_CONFIG")
        print(f"   Frame size: {analyzer.frame_size_ms}ms")
        print(f"   Min speech: {analyzer.min_speech_duration_ms}ms")
        print(f"   Min silence: {analyzer.min_silence_duration_ms}ms")
    except Exception as e:
        print(f"❌ WaveformAnalyzer initialization failed: {e}")
        return False
    
    # Test 3: Test EnhancedTranscriber with precision mode
    try:
        from src.enhanced_transcriber import EnhancedTranscriber
        import json
        
        with open('configs/precision_waveform_test.json', 'r') as f:
            config = json.load(f)
        
        transcriber = EnhancedTranscriber(config)
        print("✅ EnhancedTranscriber initialized with precision config")
        print(f"   Segmentation mode: {config['segmentation_mode']}")
    except Exception as e:
        print(f"❌ EnhancedTranscriber initialization failed: {e}")
        return False
    
    # Test 4: Check available audio files
    test_files = ['TestFile_cut.mp4.wav', 'interview_cut.mp3', 'interview.mp3']
    available = [f for f in test_files if os.path.exists(f)]
    print(f"📁 Available test files: {available}")
    
    if available:
        print("🎯 Ready to test precision waveform analysis!")
        print(f"   Recommended test: Use '{available[0]}'")
    else:
        print("⚠️ No test audio files found")
    
    print("\n🎉 All precision waveform components ready!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
