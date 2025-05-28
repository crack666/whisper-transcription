#!/usr/bin/env python3
"""
ULTIMATE BUG FIX VERIFICATION

Tests the fixed segmentation logic with multiple safety checks.
"""

import sys
import time
import os
sys.path.append('src')

from enhanced_transcriber import EnhancedAudioTranscriber
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_segmentation():
    """Test the fixed segmentation with comprehensive monitoring."""
    print("🔧 ULTIMATE BUG FIX VERIFICATION")
    print("=" * 60)
    
    # Use smaller test file first
    audio_file = "interview_cut.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    print(f"🎯 Testing fixed segmentation with: {audio_file}")
    
    # Test with very strict monitoring
    start_time = time.time()
    
    try:
        # Initialize with defensive silence + monitoring
        transcriber = EnhancedAudioTranscriber(
            model_name="base",
            language="german",
            device="cpu",
            config={
                "segmentation_mode": "defensive_silence",
                "min_silence_len": 2000,
                "padding": 1500,
                "silence_adjustment": 5.0,
                "max_segment_length": 60000,  # Smaller for testing
                "min_segment_length": 2000,
                "cleanup_segments": True
            }
        )
        
        print("\n🔄 Running transcription with bug fixes...")
        
        # Monitor segment files before transcription
        segment_files_before = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'segment' in file and file.endswith('.wav'):
                    segment_files_before.append(os.path.join(root, file))
        
        print(f"📁 Segment files before: {len(segment_files_before)}")
        
        # Run transcription
        result = transcriber.transcribe_audio_file_enhanced(audio_file)
        
        processing_time = time.time() - start_time
        
        # Monitor segment files after transcription
        segment_files_after = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'segment' in file and file.endswith('.wav'):
                    segment_files_after.append(os.path.join(root, file))
        
        print(f"📁 Segment files after: {len(segment_files_after)}")
        
        # Check for runaway segment generation
        new_segments = len(segment_files_after) - len(segment_files_before)
        
        print(f"\n✅ BUG FIX VERIFICATION RESULTS:")
        print("-" * 50)
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"📊 New segment files created: {new_segments}")
        print(f"📝 Result segments: {len(result.get('segments', []))}")
        print(f"📝 Words: {len(result.get('full_text', '').split())}")
        
        # VERIFICATION CHECKS
        success = True
        
        if new_segments > 100:
            print(f"❌ POTENTIAL BUG: Too many segment files ({new_segments})")
            success = False
        else:
            print(f"✅ Segment count OK: {new_segments} files")
        
        if processing_time > 300:  # 5 minutes
            print(f"❌ POTENTIAL BUG: Processing too slow ({processing_time:.1f}s)")
            success = False
        else:
            print(f"✅ Processing time OK: {processing_time:.1f}s")
        
        if len(result.get('segments', [])) > 50:
            print(f"❌ POTENTIAL BUG: Too many result segments ({len(result.get('segments', []))})")
            success = False
        else:
            print(f"✅ Result segment count OK: {len(result.get('segments', []))}")
        
        # Show sample output
        sample_text = result.get("full_text", "")[:200] + "..." if len(result.get("full_text", "")) > 200 else result.get("full_text", "")
        print(f"\n📖 Sample transcript:\n{sample_text}")
        
        # Cleanup test segment files
        for file in segment_files_after:
            if file not in segment_files_before:
                try:
                    os.remove(file)
                    print(f"🧹 Cleaned up: {file}")
                except:
                    pass
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_segmentation()
    if success:
        print("\n🎉 BUG FIX VERIFICATION: PASSED")
        print("The infinite loop bug appears to be fixed!")
    else:
        print("\n💥 BUG FIX VERIFICATION: FAILED")
        print("The bug may still be present!")
