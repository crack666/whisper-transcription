#!/usr/bin/env python3
"""
Test the improved adaptive segmentation mode with defensive silence principles.
Compares the new adaptive mode against defensive silence and fixed-time.
"""

import sys
import os
import time
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.enhanced_transcriber import EnhancedAudioTranscriber

def test_adaptive_mode():
    """Test the improved adaptive mode."""
    
    audio_file = "interview_cut.mp3"
    if not os.path.exists(audio_file):
        print(f"❌ Error: {audio_file} not found!")
        return
    
    print("🔄 Testing Improved Adaptive Segmentation Mode")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {
            "name": "Improved Adaptive",
            "mode": "adaptive",
            "description": "Defensive silence + speaker adaptation + no overlaps"
        },
        {
            "name": "Defensive Silence",
            "mode": "defensive_silence", 
            "description": "Conservative silence detection only"
        },
        {
            "name": "Fixed Time 30s",
            "mode": "fixed_time",
            "description": "30-second segments with overlap"
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   {config['description']}")
        print("-" * 40)
        
        # Create transcriber with specific mode
        transcriber = EnhancedAudioTranscriber(
            model_name="small",
            language="german",
            config={
                "segmentation_mode": config["mode"],
                "fixed_time_duration": 30000,
                "fixed_time_overlap": 3000
            }
        )
        
        start_time = time.time()
        
        try:
            result = transcriber.transcribe_audio_file_enhanced(audio_file)
            
            processing_time = time.time() - start_time
            
            # Extract metrics
            segments = result.get("segments", [])
            full_text = result.get("full_text", "")
            word_count = len(full_text.split())
            
            # Count successful segments
            successful_segments = [s for s in segments if not s.get("error") and s.get("text")]
            
            # Store results
            results[config["name"]] = {
                "segments_total": len(segments),
                "segments_successful": len(successful_segments),
                "word_count": word_count,
                "processing_time": processing_time,
                "full_text": full_text,
                "mode": config["mode"]
            }
            
            print(f"✅ Success:")
            print(f"   Segments: {len(successful_segments)}/{len(segments)}")
            print(f"   Words: {word_count}")
            print(f"   Time: {processing_time:.1f}s")
            print(f"   Quality: {len(successful_segments)/len(segments)*100:.1f}% success rate")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results[config["name"]] = {
                "error": str(e),
                "mode": config["mode"]
            }
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    for name, result in results.items():
        if "error" not in result:
            print(f"\n{name}:")
            print(f"  Segments: {result['segments_successful']} ({result['segments_total']} total)")
            print(f"  Words: {result['word_count']}")
            print(f"  Time: {result['processing_time']:.1f}s")
            print(f"  Speed: {result['word_count']/result['processing_time']:.1f} words/sec")
    
    # Identify best performer
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if valid_results:
        # Best by word count (completeness)
        best_words = max(valid_results.items(), key=lambda x: x[1]['word_count'])
        
        # Best by speed
        best_speed = max(valid_results.items(), key=lambda x: x[1]['word_count']/x[1]['processing_time'])
        
        print(f"\n🏆 BEST RESULTS:")
        print(f"   Most Complete: {best_words[0]} ({best_words[1]['word_count']} words)")
        print(f"   Fastest: {best_speed[0]} ({best_speed[1]['word_count']/best_speed[1]['processing_time']:.1f} words/sec)")
    
    # Save output files for manual comparison
    print(f"\n💾 Saving output files to comparison_outputs/")
    os.makedirs("comparison_outputs", exist_ok=True)
    
    for name, result in results.items():
        if "error" not in result:
            filename = f"comparison_outputs/improved_adaptive_{name.lower().replace(' ', '_')}_results.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== {name} Results ===\n")
                f.write(f"Mode: {result['mode']}\n")
                f.write(f"Segments: {result['segments_successful']}/{result['segments_total']}\n")
                f.write(f"Words: {result['word_count']}\n")
                f.write(f"Time: {result['processing_time']:.1f}s\n")
                f.write(f"Speed: {result['word_count']/result['processing_time']:.1f} words/sec\n\n")
                f.write("=== TRANSCRIPTION TEXT ===\n")
                f.write(result['full_text'])
            print(f"   ✅ {filename}")
    
    print(f"\n🎯 Test completed! Check comparison_outputs/ for detailed text comparison.")

if __name__ == "__main__":
    test_adaptive_mode()
