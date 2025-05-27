#!/usr/bin/env python3
"""
Quick transcription script that bypasses Enhanced Transcriber issues
Uses only the proven standard transcriber with fallback options
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.transcriber import AudioTranscriber
from src.utils import extract_audio_from_video
from src.config import WHISPER_MODELS, LANGUAGE_MAP

def setup_logging(verbose=False):
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def quick_transcribe(video_path: str, output_path: str, model: str = "large-v3", 
                    language: str = "german", verbose: bool = False):
    """Quick transcription using only standard transcriber."""
    
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    print(f"🎙️ Quick Transcription Tool")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    
    try:
        # Extract audio
        print(f"\n📹 Extracting audio...")
        audio_path = extract_audio_from_video(video_path)
        print(f"   ✅ Audio extracted: {audio_path}")
        
        # Initialize standard transcriber with conservative settings
        print(f"\n🧠 Loading Whisper model: {model}")
        
        # Conservative config for problematic audio
        conservative_config = {
            'min_silence_len': 1000,      # Standard 1 second
            'padding': 1000,              # 1 second padding
            'silence_adjustment': 5.0,    # More tolerant
            'cleanup_segments': True
        }
        
        transcriber = AudioTranscriber(
            model_name=model,
            language=language,
            config=conservative_config
        )
        
        print(f"   ✅ Model loaded")
        
        # Transcribe
        print(f"\n🎤 Transcribing audio...")
        result = transcriber.transcribe_audio_file(audio_path)
        
        # Check results
        segments = result.get('segments', [])
        full_text = result.get('full_text', '')
        
        print(f"\n📊 Results:")
        print(f"   Segments: {len(segments)}")
        print(f"   Characters: {len(full_text)}")
        print(f"   Words: {len(full_text.split()) if full_text else 0}")
        
        if not full_text:
            print(f"\n⚠️ No text transcribed! Trying fallback options...")
            
            # Fallback 1: More aggressive silence detection
            print(f"   🔄 Trying more aggressive silence detection...")
            fallback_config = {
                'min_silence_len': 500,       # Half second
                'padding': 1500,              # More padding
                'silence_adjustment': 8.0,    # Very tolerant
                'cleanup_segments': True
            }
            
            transcriber_fallback = AudioTranscriber(
                model_name=model,
                language=language,
                config=fallback_config
            )
            
            result = transcriber_fallback.transcribe_audio_file(audio_path)
            full_text = result.get('full_text', '')
            segments = result.get('segments', [])
            
            print(f"      Fallback segments: {len(segments)}")
            print(f"      Fallback text length: {len(full_text)}")
        
        if not full_text:
            # Fallback 2: Smaller model
            print(f"   🔄 Trying smaller model (medium)...")
            transcriber_small = AudioTranscriber(
                model_name="medium",
                language=language,
                config=conservative_config
            )
            
            result = transcriber_small.transcribe_audio_file(audio_path)
            full_text = result.get('full_text', '')
            segments = result.get('segments', [])
            
            print(f"      Small model segments: {len(segments)}")
            print(f"      Small model text length: {len(full_text)}")
        
        # Save results
        if full_text:
            print(f"\n💾 Saving results...")
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save text file
            txt_path = str(output_path).replace('.json', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Save JSON
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Text saved: {txt_path}")
            print(f"   ✅ Data saved: {output_path}")
            
            # Show preview
            print(f"\n📄 Preview (first 300 chars):")
            print(f"   '{full_text[:300]}...'")
            
        else:
            print(f"\n❌ No transcription could be generated!")
            print(f"   Possible issues:")
            print(f"   - Audio is silent or very quiet")
            print(f"   - Audio format not supported")
            print(f"   - Language detection failed")
            print(f"   - Whisper model issue")
            
            return False
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"   🧹 Cleaned up audio file")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick transcription tool (bypasses Enhanced Transcriber)")
    
    parser.add_argument("--input", type=str, required=True, help="Video file to transcribe")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--model", type=str, default="large-v3", choices=WHISPER_MODELS, 
                       help="Whisper model to use")
    parser.add_argument("--language", type=str, default="german", 
                       choices=list(LANGUAGE_MAP.keys()), help="Language")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = f"./quick_transcription_{input_path.stem}.json"
    
    # Run transcription
    success = quick_transcribe(
        video_path=args.input,
        output_path=args.output,
        model=args.model,
        language=args.language,
        verbose=args.verbose
    )
    
    if success:
        print(f"\n✅ Quick transcription completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Quick transcription failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()