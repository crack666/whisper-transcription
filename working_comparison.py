#!/usr/bin/env python3
"""
Create working comparison outputs
"""

import os
import sys
import time
import json
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.enhanced_transcriber import EnhancedAudioTranscriber

def save_transcription_comparison(method_name, results, output_dir):
    """Save detailed transcription for comparison."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    segments = results.get('segments', [])
    # Handle different possible text keys
    full_text = results.get('text', results.get('full_text', ''))
    
    # Create readable filename
    filename = f"{method_name.lower().replace(' ', '_')}_results.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"  {method_name.upper()} - TRANSCRIPTION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Statistics
        f.write("📊 STATISTICS:\n")
        f.write(f"   Segments: {len(segments)}\n")
        f.write(f"   Words: {len(full_text.split())}\n")
        f.write(f"   Characters: {len(full_text)}\n")
        f.write(f"   Processing Time: {results.get('processing_time', 0):.1f}s\n\n")
        
        # Complete text for easy reading
        f.write("📖 COMPLETE TEXT:\n")
        f.write("-" * 50 + "\n")
        f.write(full_text)
        f.write("\n\n")
        
        # Detailed segments with timestamps
        f.write("🕒 SEGMENTS WITH TIMESTAMPS:\n")
        f.write("-" * 50 + "\n")
        
        for i, segment in enumerate(segments):
            start = segment.get('start_time', 0) / 1000
            end = segment.get('end_time', 0) / 1000
            text = segment.get('text', '').strip()
            
            f.write(f"\nSegment {i+1} ({start:.1f}s - {end:.1f}s):\n")
            f.write(f'"{text}"\n')
    
    print(f"💾 Saved: {filepath}")
    return filepath

def run_single_test(audio_file, method_name, config):
    """Run a single transcription test."""
    
    print(f"\n🔧 Testing: {method_name}")
    print(f"   Config: {config}")
    
    try:
        # Use correct constructor with config
        transcriber = EnhancedAudioTranscriber(
            model_name='tiny',
            language='german',
            config=config  # Pass config to constructor
        )
        
        start_time = time.time()
        
        # Call without additional config parameters
        results = transcriber.transcribe_audio_file_enhanced(audio_file)
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        word_count = len(results.get('text', results.get('full_text', '')).split())
        print(f"   ✅ Success: {len(results.get('segments', []))} segments, {word_count} words, {processing_time:.1f}s")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("📝 TRANSCRIPTION COMPARISON - WORKING VERSION")
    print("=" * 60)
    
    audio_file = "interview_cut.mp3"
    output_dir = "comparison_outputs"
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
      # Test methods that actually work
    methods = [
        {
            "name": "Defensive Silence",
            "config": {"segmentation_mode": "defensive_silence"}
        },
        {
            "name": "Fixed Time 30s", 
            "config": {
                "segmentation_mode": "fixed_time",
                "fixed_time_duration": 30000,  # 30 seconds in ms
                "fixed_time_overlap": 2000     # 2 seconds overlap in ms
            }
        }    ]
    
    saved_files = []
    comparison_data = {}
    
    for method in methods:
        results = run_single_test(audio_file, method["name"], method["config"])
        
        if results:
            filepath = save_transcription_comparison(method["name"], results, output_dir)
            saved_files.append(filepath)
            
            # Handle different possible text keys
            full_text = results.get('text', results.get('full_text', ''))
            comparison_data[method["name"]] = {
                'segments': len(results.get('segments', [])),
                'words': len(full_text.split()),
                'time': results['processing_time'],
                'file': filepath
            }
    
    # Create comparison summary
    summary_file = os.path.join(output_dir, "COMPARISON_SUMMARY.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  TRANSCRIPTION COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Audio File: {audio_file}\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("📊 QUICK STATS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Method':<20} {'Segments':<8} {'Words':<8} {'Time':<8}\n")
        f.write("-" * 40 + "\n")
        
        for name, data in comparison_data.items():
            f.write(f"{name:<20} {data['segments']:<8} {data['words']:<8} {data['time']:<8.1f}\n")
        
        f.write("\n📁 DETAILED FILES:\n")
        f.write("-" * 40 + "\n")
        for name, data in comparison_data.items():
            f.write(f"{name}: {data['file']}\n")
        
        f.write("\n📖 HOW TO COMPARE:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Open the result files in separate editor tabs\n")
        f.write("2. Compare the 'COMPLETE TEXT' sections\n")
        f.write("3. Look for differences in:\n")
        f.write("   - Word accuracy\n")
        f.write("   - Missing or extra words\n")
        f.write("   - Natural segment breaks\n")
        f.write("   - Overall readability\n")
    
    print(f"\n✅ COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}/")
    print(f"📄 Summary: {summary_file}")
    
    if saved_files:
        print(f"\n📖 Files to compare manually:")
        for filepath in saved_files:
            print(f"   - {filepath}")
        
        print(f"\n💡 Open these files in your editor to compare the transcription quality!")

if __name__ == "__main__":
    main()
