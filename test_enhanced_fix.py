#!/usr/bin/env python3
"""
Test script to verify that the Enhanced Transcriber fix works
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.enhanced_transcriber import EnhancedAudioTranscriber
from src.utils import extract_audio_from_video

def test_enhanced_fix(video_path: str):
    """Test the fixed Enhanced Transcriber."""
    
    print(f"🔧 Testing Enhanced Transcriber Fix")
    print(f"Video: {video_path}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Extract audio
        print(f"\n📹 Extracting audio...")
        audio_path = extract_audio_from_video(video_path)
        
        # Test with fixed Enhanced Transcriber
        print(f"\n🚀 Testing fixed Enhanced Transcriber...")
        transcriber = EnhancedAudioTranscriber(
            model_name="tiny",  # Use fast model for testing
            language="german"
        )
        
        # Show the current config
        print(f"   Config min_segment_length: {transcriber.config['min_segment_length']}ms")
        print(f"   Config min_silence_len: {transcriber.config['min_silence_len']}ms")
        
        # Run enhanced transcription
        result = transcriber.transcribe_audio_file_enhanced(audio_path)
        
        # Check results
        segments = result.get('segments', [])
        full_text = result.get('full_text', '')
        
        print(f"\n📊 Results:")
        print(f"   Segments: {len(segments)}")
        print(f"   Characters: {len(full_text)}")
        print(f"   Words: {len(full_text.split()) if full_text else 0}")
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            return False
        
        if full_text:
            print(f"   ✅ SUCCESS: Text transcribed!")
            print(f"   Preview: '{full_text[:200]}...'")
            
            # Show some segments
            print(f"\n📋 First few segments:")
            for i, segment in enumerate(segments[:3]):
                if segment.get('text'):
                    print(f"   {i+1}: {segment['text'][:100]}...")
            
            return True
        else:
            print(f"   ❌ FAIL: No text transcribed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    if len(sys.argv) != 2:
        print("Usage: python test_enhanced_fix.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    success = test_enhanced_fix(video_path)
    
    if success:
        print(f"\n✅ Enhanced Transcriber fix SUCCESSFUL!")
        print(f"You can now use:")
        print(f"   python study_processor_v2.py --input \"{video_path}\" --config configs/lecture_optimized.json")
    else:
        print(f"\n❌ Enhanced Transcriber fix FAILED!")
        print(f"Use fallback:")
        print(f"   python quick_transcribe.py --input \"{video_path}\" --verbose")

if __name__ == "__main__":
    main()