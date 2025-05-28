#!/usr/bin/env python3
"""
Test the new defensive silence detection approach
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.enhanced_transcriber import EnhancedAudioTranscriber

def save_text_to_file(method_name, full_text, segments, word_count, char_count, processing_time):
    """Save transcription text to file for comparison."""
    import os
    
    output_dir = "text_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{method_name.lower().replace(' ', '_')}_transcription.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== {method_name.upper()} TRANSCRIPTION ===\n")
        f.write(f"Segments: {len(segments)}\n")
        f.write(f"Words: {word_count}\n")
        f.write(f"Characters: {char_count}\n")
        f.write(f"Processing time: {processing_time:.1f}s\n\n")
        
        f.write("FULL TEXT:\n")
        f.write(full_text)
        f.write("\n\n")
        
        f.write("SEGMENTS:\n")
        for i, segment in enumerate(segments):
            start_time = segment.get('start_time', 0) / 1000  # Convert to seconds
            end_time = segment.get('end_time', 0) / 1000
            text = segment.get('text', '')
            f.write(f"\nSegment {i+1} ({start_time:.1f}s - {end_time:.1f}s):\n")
            f.write(text)
            f.write("\n")
    
    print(f"💾 Saved {method_name} text to: {filepath}")
    return filepath

def test_defensive_silence():
    """Test defensive silence detection vs other methods."""
    
    print("🛡️ TESTING DEFENSIVE SILENCE DETECTION")
    print("=" * 60)
    
    # Test with interview_cut.mp3
    audio_file = "interview_cut.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Test configurations
    test_configs = [
        {
            "name": "Defensive Silence",
            "config": {
                "segmentation_mode": "defensive_silence",
                "min_silence_len": 2000  # 2 seconds minimum
            }
        },
        {
            "name": "Current Enhanced",
            "config": {
                "segmentation_mode": "silence_detection",
                "min_silence_len": 2000,
                "silence_adjustment": 5.0
            }
        },
        {
            "name": "Fixed-Time 30s",
            "config": {
                "segmentation_mode": "fixed_time",
                "fixed_time_duration": 30000,
                "fixed_time_overlap": 2000
            }
        },
        {
            "name": "Adaptive Mode",
            "config": {
                "segmentation_mode": "adaptive"
            }
        }
    ]
    
    results = []
    
    for test in test_configs:
        print(f"\n🔧 Testing {test['name']}...")
        
        try:
            transcriber = EnhancedAudioTranscriber(
                model_name="tiny",
                language="german", 
                config=test['config']
            )
            
            result = transcriber.transcribe_audio_file_enhanced(audio_file)
            
            segments = result.get('segments', [])
            full_text = result.get('full_text', '')
            processing_time = result.get('processing_time_seconds', 0)
            
            word_count = len(full_text.split())
            char_count = len(full_text)
            
            print(f"   📊 Results:")
            print(f"      Segments: {len(segments)}")
            print(f"      Words: {word_count}")
            print(f"      Characters: {char_count}")
            print(f"      Processing time: {processing_time:.1f}s")
            
            # Show segment timing
            if segments:
                print(f"      First segment: {segments[0].get('start_time', 0)/1000:.1f}s")
                print(f"      Last segment: {segments[-1].get('end_time', 0)/1000:.1f}s")
                
                # Show first few segments
                print(f"      Sample segments:")
                for i, seg in enumerate(segments[:3]):
                    start = seg.get('start_time', 0) / 1000
                    end = seg.get('end_time', 0) / 1000
                    text_preview = seg.get('text', '')[:50] + ('...' if len(seg.get('text', '')) > 50 else '')
                    print(f"        {i+1}: {start:.1f}s-{end:.1f}s '{text_preview}'")
            
            results.append({
                "name": test['name'],
                "segments": len(segments),
                "words": word_count,
                "characters": char_count,
                "processing_time": processing_time,
                "config": test['config']
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if results:
        print(f"\n📋 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Method':<20} {'Segments':<10} {'Words':<8} {'Chars':<8} {'Time':<8}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['name']:<20} {r['segments']:<10} {r['words']:<8} {r['characters']:<8} {r['processing_time']:<8.1f}")
        
        # Find best performing
        best_words = max(results, key=lambda x: x['words'])
        fastest = min(results, key=lambda x: x['processing_time'])
        
        print(f"\n🏆 Performance Leaders:")
        print(f"   Most words: {best_words['name']} ({best_words['words']} words)")
        print(f"   Fastest: {fastest['name']} ({fastest['processing_time']:.1f}s)")

def test_defensive_with_full_interview():
    """Test defensive silence on full interview.mp3"""
    
    print(f"\n🎙️ TESTING DEFENSIVE SILENCE ON FULL INTERVIEW")
    print("=" * 60)
    
    audio_file = "interview.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    config = {
        "segmentation_mode": "defensive_silence",
        "min_silence_len": 2000  # Conservative 2 seconds
    }
    
    try:
        print("🔄 Starting defensive silence transcription...")
        
        transcriber = EnhancedAudioTranscriber(
            model_name="tiny",
            language="german", 
            config=config
        )
        
        result = transcriber.transcribe_audio_file_enhanced(audio_file)
        
        segments = result.get('segments', [])
        full_text = result.get('full_text', '')
        processing_time = result.get('processing_time_seconds', 0)
        
        print(f"\n✅ FULL INTERVIEW RESULTS:")
        print(f"   Total segments: {len(segments)}")
        print(f"   Total words: {len(full_text.split())}")
        print(f"   Total characters: {len(full_text)}")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Audio duration: ~29 minutes")
        
        if segments:
            print(f"\n📝 Sample segment texts:")
            for i, seg in enumerate(segments[:5]):
                text = seg.get('text', '').strip()
                start = seg.get('start_time', 0) / 1000 / 60  # minutes
                end = seg.get('end_time', 0) / 1000 / 60
                print(f"   Segment {i+1} ({start:.1f}-{end:.1f}min): '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Compare with our previous best result (fixed-time 30s with 2s overlap: 332 words)
        previous_best = 332
        improvement = len(full_text.split()) / previous_best * 100 if previous_best > 0 else 0
        
        print(f"\n📈 Comparison with previous best (Fixed-Time 30s):")
        print(f"   Previous: {previous_best} words")
        print(f"   Defensive: {len(full_text.split())} words")
        print(f"   Performance: {improvement:.1f}% of previous best")
        
    except Exception as e:
        print(f"❌ Error testing full interview: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test on short clip first
    test_defensive_silence()
    
    # Ask user if they want to test full interview
    response = input(f"\n❓ Test defensive silence on full interview.mp3? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        test_defensive_with_full_interview()
    else:
        print("Skipping full interview test.")
