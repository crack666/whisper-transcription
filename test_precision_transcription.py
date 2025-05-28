#!/usr/bin/env python3
"""
Test precision waveform transcription on actual audio.
"""
import sys
import os
import json
sys.path.append('.')

def main():
    print("🔬 PRECISION WAVEFORM TRANSCRIPTION TEST")
    print("=" * 50)
    
    # Load precision waveform configuration
    with open('configs/precision_waveform_test.json', 'r') as f:
        config = json.load(f)
    
    print(f"📋 Loaded config with segmentation mode: {config['segmentation_mode']}")
    
    # Choose test file
    test_file = "interview_cut.mp3"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    file_size = os.path.getsize(test_file) / (1024*1024)
    print(f"📁 Test file: {test_file} ({file_size:.1f} MB)")
    
    # Import and initialize transcriber
    from src.enhanced_transcriber import EnhancedAudioTranscriber
    transcriber = EnhancedAudioTranscriber(config=config)
    print("✅ EnhancedAudioTranscriber initialized with precision waveform mode")
    
    # Run transcription
    print("\n🎯 Starting precision waveform transcription...")
    try:
        result = transcriber.transcribe(test_file)
        
        # Display results
        print("\n📊 TRANSCRIPTION RESULTS:")
        print(f"   Text length: {len(result.get('text', ''))} characters")
        print(f"   Segments: {len(result.get('segments', []))}")
        
        # Show segment info
        segments = result.get('segments', [])
        if segments:
            print(f"   First segment: {segments[0].get('start', 0):.1f}s - {segments[0].get('end', 0):.1f}s")
            print(f"   Last segment: {segments[-1].get('start', 0):.1f}s - {segments[-1].get('end', 0):.1f}s")
        
        # Show partial text
        text = result.get('text', '')
        if text:
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"   Text preview: {preview}")
        
        print("\n🎉 Precision waveform transcription completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n📊 Final result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
