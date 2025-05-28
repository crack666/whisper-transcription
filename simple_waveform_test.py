#!/usr/bin/env python3
"""
Simple Waveform Test - Direct Import Test
"""

import sys
import traceback

# Add src to path
sys.path.insert(0, 'src')

print("🔬 SIMPLE WAVEFORM IMPORT TEST")
print("=" * 40)

try:
    print("1. Testing WaveformAnalyzer import...")
    from waveform_analyzer import WaveformAnalyzer
    print("✅ WaveformAnalyzer imported successfully!")
    
    print("2. Creating WaveformAnalyzer instance...")
    analyzer = WaveformAnalyzer()
    print("✅ WaveformAnalyzer instance created!")
    
    print("3. Testing audio file analysis...")
    import os
    if os.path.exists("interview_cut.mp3"):
        print("   Found interview_cut.mp3, analyzing...")
        segments = analyzer.analyze_speech_segments("interview_cut.mp3")
        print(f"✅ Analysis complete! Found {len(segments)} segments")
        
        if segments:
            print(f"   First segment: {segments[0][0]/1000:.1f}s - {segments[0][1]/1000:.1f}s")
            print(f"   Last segment: {segments[-1][0]/1000:.1f}s - {segments[-1][1]/1000:.1f}s")
    else:
        print("   ⚠️  interview_cut.mp3 not found, skipping audio analysis")
    
    print("\n🎉 ALL TESTS PASSED!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ General Error: {e}")
    traceback.print_exc()
