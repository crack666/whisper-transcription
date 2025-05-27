#!/usr/bin/env python3
"""
Test the fixed Enhanced Transcriber with the lecture_fixed.json config
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_enhanced_transcriber():
    """Test the fixed Enhanced Transcriber"""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🔧 Testing Fixed Enhanced Transcriber")
    print("=" * 50)
    
    try:
        # Import with proper error handling
        try:
            from src.enhanced_transcriber import EnhancedAudioTranscriber
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   This likely means whisper or other dependencies are not installed")
            return False
        
        # Load config
        config_path = "configs/lecture_fixed.json"
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded config: {config_path}")
        transcription_config = config.get('transcription', {})
        print(f"   min_silence_len: {transcription_config.get('min_silence_len')}ms")
        print(f"   silence_adjustment: {transcription_config.get('silence_adjustment')}")
        print(f"   min_segment_length: {transcription_config.get('min_segment_length')}ms")
        
        # Test audio file
        test_audio = "interview.mp3"
        if not os.path.exists(test_audio):
            print(f"❌ Test audio not found: {test_audio}")
            return False
        
        print(f"✅ Test audio: {test_audio}")
        
        # Initialize Enhanced Transcriber
        print(f"\n🧠 Initializing Enhanced Transcriber...")
        try:
            transcriber = EnhancedAudioTranscriber(
                model_name=transcription_config.get('model', 'large-v3'),
                language=transcription_config.get('language', 'german'),
                config=transcription_config
            )
            print(f"   ✅ Enhanced Transcriber initialized")
        except Exception as e:
            print(f"   ❌ Failed to initialize Enhanced Transcriber: {e}")
            return False
        
        # Test transcription
        print(f"\n🎤 Testing Enhanced Transcription...")
        try:
            result = transcriber.transcribe_audio_file_enhanced(test_audio)
            
            # Check results
            segments = result.get('segments', [])
            full_text = result.get('full_text', '')
            
            print(f"\n📊 Enhanced Transcriber Results:")
            print(f"   Segments: {len(segments)}")
            print(f"   Characters: {len(full_text)}")
            print(f"   Words: {len(full_text.split()) if full_text else 0}")
            
            if full_text:
                print(f"   ✅ SUCCESS: Enhanced Transcriber working!")
                print(f"   Preview: '{full_text[:100]}...'")
                return True
            else:
                print(f"   ❌ FAILED: No text transcribed by Enhanced Transcriber")
                return False
                
        except Exception as e:
            print(f"   ❌ Enhanced Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_transcriber()
    
    if success:
        print(f"\n🎉 Enhanced Transcriber fix SUCCESSFUL!")
        print(f"   ✅ Ready for production use with large-v3 model")
        sys.exit(0)
    else:
        print(f"\n💥 Enhanced Transcriber fix FAILED!")
        print(f"   ❌ Additional debugging needed")
        sys.exit(1)