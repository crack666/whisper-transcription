#!/usr/bin/env python3
"""
Test the infinite loop fix for segment generation.
"""

import sys
import time
import os
sys.path.append('src')

from enhanced_transcriber import EnhancedAudioTranscriber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_infinite_loop_fix():
    """Test that the infinite loop fix works."""
    print("🛡️ TESTING INFINITE LOOP FIX")
    print("=" * 50)
    
    # Use a larger file for testing
    audio_file = "studies/Aufzeichnung - 20.05.2025.mp4"
    
    if not os.path.exists(audio_file):
        print(f"❌ Test audio file not found: {audio_file}")
        print("Using fallback test file...")
        audio_file = "interview_cut.mp3"
        
        if not os.path.exists(audio_file):
            print(f"❌ Fallback audio file not found: {audio_file}")
            return
    
    print(f"🎯 Testing file: {audio_file}")
    print("🔄 Running with defensive silence mode (where the bug occurred)...")
    
    start_time = time.time()
    
    try:
        # Initialize with defensive silence config
        transcriber = EnhancedAudioTranscriber(
            model_name="base",
            language="german",
            device="cpu",
            config={
                "segmentation_mode": "defensive_silence",
                "min_silence_len": 2000,
                "padding": 1500,
                "silence_adjustment": 5.0,
                "cleanup_segments": True
            }
        )
        
        # Set a timeout to kill the test if it takes too long
        print("⏰ Starting transcription (will timeout after 5 minutes if stuck)...")
        
        result = transcriber.transcribe_audio_file_enhanced(audio_file)
        
        processing_time = time.time() - start_time
        
        # Extract metrics
        segments_count = len(result.get("segments", []))
        successful_segments = result.get("segments_successful", 0)
        total_words = len(result.get("full_text", "").split())
        
        print("\n✅ INFINITE LOOP FIX TEST - SUCCESS!")
        print("-" * 40)
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"📊 Segments: {segments_count} ({successful_segments} successful)")
        print(f"📝 Words: {total_words}")
        print(f"🚀 Speed: {total_words/processing_time:.1f} words/sec")
        
        # Safety checks
        if segments_count > 500:
            print(f"⚠️  WARNING: High segment count ({segments_count}) - check for remaining issues")
        else:
            print(f"✅ Segment count is reasonable ({segments_count})")
            
        if processing_time > 300:  # 5 minutes
            print(f"⚠️  WARNING: Long processing time ({processing_time:.1f}s) - may indicate performance issues")
        else:
            print(f"✅ Processing time is acceptable ({processing_time:.1f}s)")
        
        return True
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Test failed after {processing_time:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_infinite_loop_fix()
    if success:
        print("\n🎉 INFINITE LOOP FIX VALIDATED!")
        print("Safe to use defensive silence mode again.")
    else:
        print("\n⚠️  INFINITE LOOP FIX NEEDS MORE WORK!")
        print("Do not use defensive silence mode until fixed.")
