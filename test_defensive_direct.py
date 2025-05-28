#!/usr/bin/env python3
"""
Direct Defensive Silence Test

Simple test of defensive silence mode using the enhanced transcriber directly.
"""

import sys
import time
import os
sys.path.append('src')

from enhanced_transcriber import EnhancedAudioTranscriber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_defensive_silence_direct():
    """Test defensive silence mode directly."""
    print("🛡️ DIRECT DEFENSIVE SILENCE TEST")
    print("=" * 50)
    
    # Use the interview_cut.mp3 for testing
    audio_file = "interview_cut.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"🎯 Testing file: {audio_file}")
    
    # Test defensive silence mode
    print("\n🔄 Testing DEFENSIVE SILENCE mode...")
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
        
        # Run transcription
        result = transcriber.transcribe_audio_file_enhanced(audio_file)
        
        processing_time = time.time() - start_time
        
        # Extract metrics
        segments_count = len(result.get("segments", []))
        successful_segments = result.get("segments_successful", 0)
        total_words = len(result.get("full_text", "").split())
        audio_duration = result.get("total_duration", 0) / 1000.0
        
        print("\n✅ DEFENSIVE SILENCE RESULTS:")
        print("-" * 40)
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"🎵 Audio duration: {audio_duration:.1f}s")
        print(f"📊 Segments: {segments_count} ({successful_segments} successful)")
        print(f"📝 Words: {total_words}")
        print(f"🚀 Speed: {total_words/processing_time:.1f} words/sec")
        print(f"⚡ Real-time factor: {audio_duration/processing_time:.1f}x")
        
        # Save results
        os.makedirs("defensive_direct_results", exist_ok=True)
        
        with open("defensive_direct_results/transcript.txt", "w", encoding="utf-8") as f:
            f.write("=== DEFENSIVE SILENCE MODE TRANSCRIPT ===\n\n")
            f.write(result.get("full_text", ""))
            f.write(f"\n\n=== STATISTICS ===\n")
            f.write(f"Processing Time: {processing_time:.1f}s\n")
            f.write(f"Audio Duration: {audio_duration:.1f}s\n")
            f.write(f"Segments: {segments_count}\n")
            f.write(f"Words: {total_words}\n")
            f.write(f"Speed: {total_words/processing_time:.1f} words/sec\n")
            f.write(f"Real-time factor: {audio_duration/processing_time:.1f}x\n")
        
        print(f"\n💾 Results saved to: defensive_direct_results/transcript.txt")
        
        # Show sample text
        sample_text = result.get("full_text", "")[:200] + "..." if len(result.get("full_text", "")) > 200 else result.get("full_text", "")
        print(f"\n📖 Sample transcript:\n{sample_text}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_defensive_silence_direct()
