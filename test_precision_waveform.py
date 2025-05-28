#!/usr/bin/env python3
"""
Precision Waveform Test

Test the new scientific waveform analyzer for precise speech segmentation.
This uses mathematical analysis instead of heuristic silence detection.
"""

import sys
import time
import os
sys.path.append('src')

from enhanced_transcriber import EnhancedAudioTranscriber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_precision_waveform():
    """Test precision waveform mode."""
    print("🔬 PRECISION WAVEFORM TEST")
    print("=" * 50)
    
    # Use the interview_cut.mp3 for initial testing
    audio_file = "interview_cut.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"🎯 Testing file: {audio_file}")
    
    # Test precision waveform mode
    print("\n🔬 Testing PRECISION WAVEFORM mode...")
    start_time = time.time()
    
    try:
        # Initialize with precision waveform config
        transcriber = EnhancedAudioTranscriber(
            model_name="base",
            language="german",
            device="cpu",
            config={
                "segmentation_mode": "precision_waveform",
                "min_silence_len": 1000,
                "padding": 1500,
                "silence_adjustment": 3.0,
                "cleanup_segments": True,
                "create_waveform_visualization": True
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
        
        print("\n✅ PRECISION WAVEFORM RESULTS:")
        print("-" * 40)
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"🎵 Audio duration: {audio_duration:.1f}s")
        print(f"📊 Segments: {segments_count} ({successful_segments} successful)")
        print(f"📝 Words: {total_words}")
        print(f"🚀 Speed: {total_words/processing_time:.1f} words/sec")
        print(f"⚡ Real-time factor: {audio_duration/processing_time:.1f}x")
        
        # Save results
        os.makedirs("precision_waveform_results", exist_ok=True)
        
        with open("precision_waveform_results/transcript.txt", "w", encoding="utf-8") as f:
            f.write("=== PRECISION WAVEFORM MODE TRANSCRIPT ===\n\n")
            f.write(result.get("full_text", ""))
            f.write(f"\n\n=== STATISTICS ===\n")
            f.write(f"Processing Time: {processing_time:.1f}s\n")
            f.write(f"Audio Duration: {audio_duration:.1f}s\n")
            f.write(f"Segments: {segments_count}\n")
            f.write(f"Words: {total_words}\n")
            f.write(f"Speed: {total_words/processing_time:.1f} words/sec\n")
            f.write(f"Real-time factor: {audio_duration/processing_time:.1f}x\n")
        
        print(f"\n💾 Results saved to: precision_waveform_results/transcript.txt")
        
        # Show sample text
        sample_text = result.get("full_text", "")[:200] + "..." if len(result.get("full_text", "")) > 200 else result.get("full_text", "")
        print(f"\n📖 Sample transcript:\n{sample_text}")
        
        # Check if visualization was created
        viz_files = [f for f in os.listdir(".") if f.endswith("_waveform_analysis.png")]
        if viz_files:
            print(f"\n📊 Waveform visualization created: {viz_files[0]}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_modes():
    """Compare different segmentation modes."""
    print("\n🔄 COMPARING SEGMENTATION MODES")
    print("=" * 50)
    
    audio_file = "interview_cut.mp3"
    modes = [
        ("defensive_silence", "🛡️ Defensive Silence"),
        ("precision_waveform", "🔬 Precision Waveform"),
        ("adaptive", "🔄 Adaptive")
    ]
    
    results = {}
    
    for mode_name, mode_label in modes:
        print(f"\n{mode_label} Mode...")
        start_time = time.time()
        
        try:
            transcriber = EnhancedAudioTranscriber(
                model_name="base",
                language="german", 
                device="cpu",
                config={
                    "segmentation_mode": mode_name,
                    "cleanup_segments": True,
                    "create_waveform_visualization": (mode_name == "precision_waveform")
                }
            )
            
            result = transcriber.transcribe_audio_file_enhanced(audio_file)
            processing_time = time.time() - start_time
            
            segments_count = len(result.get("segments", []))
            total_words = len(result.get("full_text", "").split())
            audio_duration = result.get("total_duration", 0) / 1000.0
            
            results[mode_name] = {
                "segments": segments_count,
                "words": total_words,
                "time": processing_time,
                "speed": total_words/processing_time if processing_time > 0 else 0,
                "real_time_factor": audio_duration/processing_time if processing_time > 0 else 0
            }
            
            print(f"   📊 {segments_count} segments, {total_words} words")
            print(f"   ⏱️ {processing_time:.1f}s ({total_words/processing_time:.1f} w/s)")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[mode_name] = None
    
    # Print comparison table
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 60)
    print(f"{'Mode':<20} {'Segments':<10} {'Words':<8} {'Speed':<10} {'RT Factor':<10}")
    print("-" * 60)
    
    for mode_name, mode_label in modes:
        if results.get(mode_name):
            r = results[mode_name]
            print(f"{mode_label:<20} {r['segments']:<10} {r['words']:<8} {r['speed']:<10.1f} {r['real_time_factor']:<10.1f}")
        else:
            print(f"{mode_label:<20} {'FAILED':<10}")
    
    return results

if __name__ == "__main__":
    # Test precision waveform mode
    test_precision_waveform()
    
    # Compare different modes
    compare_modes()
