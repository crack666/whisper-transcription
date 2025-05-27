#!/usr/bin/env python3
"""
Debug script for transcription issues - detailed audio analysis and testing
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import traceback

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.enhanced_transcriber import EnhancedAudioTranscriber
from src.transcriber import AudioTranscriber
from src.utils import extract_audio_from_video

def setup_debug_logging():
    """Setup detailed debug logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug_transcription.log')
        ]
    )

def analyze_audio_file(audio_path: str):
    """Detailed audio file analysis."""
    print(f"\n🔍 Analyzing audio file: {audio_path}")
    
    try:
        audio = AudioSegment.from_file(audio_path)
        
        print(f"   📊 Basic properties:")
        print(f"      Duration: {len(audio)/1000:.1f} seconds")
        print(f"      Sample rate: {audio.frame_rate} Hz")
        print(f"      Channels: {audio.channels}")
        print(f"      Frame width: {audio.frame_width}")
        print(f"      Max dBFS: {audio.max_dBFS}")
        print(f"      dBFS: {audio.dBFS}")
        
        # Check if audio is silent
        if audio.dBFS == float('-inf'):
            print(f"   ❌ Audio appears to be completely silent!")
            return False
        
        # Sample analysis - check chunks
        chunk_size = 10000  # 10 seconds
        chunks_analyzed = 0
        non_silent_chunks = 0
        
        print(f"   🔊 Volume analysis (10s chunks):")
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 1000:  # At least 1 second
                chunk_db = chunk.dBFS
                chunks_analyzed += 1
                
                if chunk_db > -60:  # Not silent
                    non_silent_chunks += 1
                
                if chunks_analyzed <= 10:  # Show first 10 chunks
                    print(f"      Chunk {chunks_analyzed}: {chunk_db:.1f} dBFS")
        
        print(f"   📈 Summary:")
        print(f"      Total chunks: {chunks_analyzed}")
        print(f"      Non-silent chunks: {non_silent_chunks}")
        print(f"      Speech ratio: {non_silent_chunks/chunks_analyzed:.1%}")
        
        return non_silent_chunks > 0
        
    except Exception as e:
        print(f"   ❌ Error analyzing audio: {e}")
        traceback.print_exc()
        return False

def test_basic_whisper(audio_path: str):
    """Test basic Whisper functionality."""
    print(f"\n🧪 Testing basic Whisper model...")
    
    try:
        import whisper
        
        # Load smallest model for quick test
        print(f"   Loading tiny model...")
        model = whisper.load_model("tiny")
        
        # Test transcription of first 30 seconds
        audio = AudioSegment.from_file(audio_path)
        test_chunk = audio[:30000]  # First 30 seconds
        
        test_file = "temp_test_chunk.wav"
        test_chunk.export(test_file, format="wav")
        
        print(f"   Transcribing 30-second test chunk...")
        result = model.transcribe(test_file, language="de", verbose=True)
        
        print(f"   📝 Result:")
        print(f"      Text: '{result['text']}'")
        print(f"      Language detected: {result.get('language', 'unknown')}")
        
        # Cleanup
        os.remove(test_file)
        
        return len(result['text'].strip()) > 0
        
    except Exception as e:
        print(f"   ❌ Basic Whisper test failed: {e}")
        traceback.print_exc()
        return False

def test_standard_transcriber(audio_path: str):
    """Test standard transcriber."""
    print(f"\n🔧 Testing standard AudioTranscriber...")
    
    try:
        transcriber = AudioTranscriber(model_name="tiny", language="german")
        result = transcriber.transcribe_audio_file(audio_path)
        
        print(f"   📊 Standard transcriber results:")
        print(f"      Segments: {len(result.get('segments', []))}")
        print(f"      Text length: {len(result.get('full_text', ''))}")
        print(f"      Success rate: {result.get('segments_successful', 0)}/{result.get('segments_total', 0)}")
        
        if result.get('full_text'):
            print(f"      First 200 chars: '{result['full_text'][:200]}...'")
            return True
        else:
            print(f"      ❌ No text transcribed")
            return False
            
    except Exception as e:
        print(f"   ❌ Standard transcriber failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_transcriber(audio_path: str):
    """Test enhanced transcriber with debug info."""
    print(f"\n🚀 Testing EnhancedAudioTranscriber...")
    
    try:
        transcriber = EnhancedAudioTranscriber(model_name="tiny", language="german")
        
        # First analyze speech patterns
        print(f"   🔍 Analyzing speech patterns...")
        analysis = transcriber.analyze_speech_patterns(audio_path)
        
        print(f"   📊 Speech analysis:")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.3f}")
            else:
                print(f"      {key}: {value}")
        
        # Try enhanced transcription
        print(f"   🎙️ Running enhanced transcription...")
        result = transcriber.transcribe_audio_file_enhanced(audio_path)
        
        print(f"   📋 Enhanced transcriber results:")
        print(f"      Segments: {len(result.get('segments', []))}")
        print(f"      Text length: {len(result.get('full_text', ''))}")
        print(f"      Success rate: {result.get('segments_successful', 0)}/{result.get('segments_total', 0)}")
        
        if 'error' in result:
            print(f"      ❌ Error: {result['error']}")
            return False
        
        if result.get('full_text'):
            print(f"      First 200 chars: '{result['full_text'][:200]}...'")
            return True
        else:
            print(f"      ❌ No text transcribed")
            return False
            
    except Exception as e:
        print(f"   ❌ Enhanced transcriber failed: {e}")
        traceback.print_exc()
        return False

def test_silence_detection(audio_path: str):
    """Test different silence detection parameters."""
    print(f"\n🔇 Testing silence detection parameters...")
    
    try:
        from pydub import silence
        audio = AudioSegment.from_file(audio_path)
        
        # Test different parameters
        test_configs = [
            {"min_silence_len": 1000, "silence_thresh": audio.dBFS + 3, "name": "Standard"},
            {"min_silence_len": 2000, "silence_thresh": audio.dBFS + 5, "name": "Conservative"},
            {"min_silence_len": 3000, "silence_thresh": audio.dBFS + 8, "name": "Very Conservative"},
            {"min_silence_len": 500, "silence_thresh": audio.dBFS + 1, "name": "Aggressive"},
        ]
        
        for config in test_configs:
            try:
                nonsilent = silence.detect_nonsilent(
                    audio,
                    min_silence_len=config["min_silence_len"],
                    silence_thresh=config["silence_thresh"]
                )
                
                total_speech_time = sum(end - start for start, end in nonsilent) / 1000.0
                coverage = total_speech_time / (len(audio) / 1000.0)
                
                print(f"   📏 {config['name']} (silence_len={config['min_silence_len']}, thresh={config['silence_thresh']:.1f}):")
                print(f"      Segments found: {len(nonsilent)}")
                print(f"      Total speech time: {total_speech_time:.1f}s")
                print(f"      Coverage: {coverage:.1%}")
                
            except Exception as e:
                print(f"   ❌ {config['name']} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Silence detection test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    if len(sys.argv) != 2:
        print("Usage: python debug_transcription.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    print(f"🐛 Transcription Debug Tool")
    print(f"Video: {video_path}")
    
    # Setup debug logging
    setup_debug_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting debug session for: {video_path}")
    
    try:
        # Extract audio
        print(f"\n🎵 Extracting audio...")
        audio_path = extract_audio_from_video(video_path)
        print(f"   Audio extracted to: {audio_path}")
        
        # Run all tests
        results = {}
        
        results['audio_analysis'] = analyze_audio_file(audio_path)
        results['basic_whisper'] = test_basic_whisper(audio_path)
        results['standard_transcriber'] = test_standard_transcriber(audio_path)
        results['enhanced_transcriber'] = test_enhanced_transcriber(audio_path)
        results['silence_detection'] = test_silence_detection(audio_path)
        
        # Summary
        print(f"\n📋 Summary:")
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if not results['audio_analysis']:
            print(f"   🔧 Audio appears to be silent or corrupted")
            print(f"      - Check if video has audio track")
            print(f"      - Try manual audio extraction with ffmpeg")
        
        elif not results['basic_whisper']:
            print(f"   🔧 Basic Whisper functionality failed")
            print(f"      - Check Whisper installation")
            print(f"      - Try different audio format")
        
        elif results['standard_transcriber'] and not results['enhanced_transcriber']:
            print(f"   🔧 Standard works but Enhanced fails")
            print(f"      - Use standard transcriber as fallback")
            print(f"      - Check enhanced transcriber parameters")
        
        elif not results['standard_transcriber'] and not results['enhanced_transcriber']:
            print(f"   🔧 Both transcribers failed")
            print(f"      - Audio might be too quiet or in unsupported format")
            print(f"      - Try audio normalization")
        
        else:
            print(f"   ✅ All tests passed - issue might be configuration-specific")
        
        print(f"\n📄 Detailed logs saved to: debug_transcription.log")
        
    except Exception as e:
        print(f"\n❌ Debug session failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()