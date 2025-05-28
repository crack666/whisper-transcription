#!/usr/bin/env python3
"""
Debug script to analyze silence detection issues with interview.mp3
"""

import os
import json
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def analyze_silence_detection(audio_file: str):
    """Comprehensive analysis of silence detection parameters"""
    
    print(f"🔍 Analyzing silence detection for: {audio_file}")
    
    # Load audio
    audio = AudioSegment.from_file(audio_file)
    duration = len(audio) / 1000.0
    print(f"📊 Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Test different silence detection parameters
    test_configs = [
        {"name": "Current Enhanced", "min_silence_len": 2000, "silence_thresh": -23},
        {"name": "More Aggressive", "min_silence_len": 1000, "silence_thresh": -20},
        {"name": "Very Aggressive", "min_silence_len": 500, "silence_thresh": -18},
        {"name": "Ultra Aggressive", "min_silence_len": 300, "silence_thresh": -15},
        {"name": "Minimal Silence", "min_silence_len": 100, "silence_thresh": -12},
        {"name": "Conservative", "min_silence_len": 3000, "silence_thresh": -25},
        {"name": "Manual Fallback", "min_silence_len": 50, "silence_thresh": -10},
    ]
    
    print("\n🎛️ Testing different silence detection parameters:")
    print("=" * 80)
    
    results = []
    
    for config in test_configs:
        try:
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=config["min_silence_len"],
                silence_thresh=config["silence_thresh"]
            )
            
            if nonsilent_ranges:
                total_speech_time = sum(end - start for start, end in nonsilent_ranges) / 1000.0
                speech_percentage = (total_speech_time / duration) * 100
                first_speech = nonsilent_ranges[0][0] / 1000.0
                last_speech = nonsilent_ranges[-1][1] / 1000.0
                
                print(f"📋 {config['name']:20} | "
                      f"Segments: {len(nonsilent_ranges):3} | "
                      f"Speech: {speech_percentage:5.1f}% | "
                      f"First: {first_speech:6.1f}s | "
                      f"Last: {last_speech:6.1f}s")
                
                results.append({
                    "config": config,
                    "segments": len(nonsilent_ranges),
                    "speech_percentage": speech_percentage,
                    "first_speech": first_speech,
                    "nonsilent_ranges": nonsilent_ranges[:5]  # First 5 for analysis
                })
            else:
                print(f"❌ {config['name']:20} | No segments detected!")
                
        except Exception as e:
            print(f"💥 {config['name']:20} | ERROR: {e}")
    
    # Show the best result details
    if results:
        best_result = max(results, key=lambda x: x["speech_percentage"])
        print(f"\n🏆 Best result: {best_result['config']['name']}")
        print(f"   📊 {best_result['segments']} segments covering {best_result['speech_percentage']:.1f}% of audio")
        print(f"   🎯 First 5 segments:")
        
        for i, (start, end) in enumerate(best_result["nonsilent_ranges"]):
            duration_seg = (end - start) / 1000.0
            print(f"      Segment {i}: {start/1000:.1f}s - {end/1000:.1f}s ({duration_seg:.1f}s)")
    
    return results

def analyze_audio_volume_distribution(audio_file: str):
    """Analyze volume distribution to understand silence thresholds"""
    
    print(f"\n🔊 Volume Analysis for: {audio_file}")
    
    audio = AudioSegment.from_file(audio_file)
    
    # Analyze volume in chunks
    chunk_size = 1000  # 1 second chunks
    volumes = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) > 0:
            volume = chunk.dBFS
            if np.isfinite(volume):
                volumes.append(volume)
    
    if volumes:
        volumes = np.array(volumes)
        
        print(f"📊 Volume Statistics:")
        print(f"   Mean: {np.mean(volumes):.1f} dB")
        print(f"   Std:  {np.std(volumes):.1f} dB")
        print(f"   Min:  {np.min(volumes):.1f} dB")
        print(f"   Max:  {np.max(volumes):.1f} dB")
        print(f"   10th percentile: {np.percentile(volumes, 10):.1f} dB")
        print(f"   25th percentile: {np.percentile(volumes, 25):.1f} dB")
        print(f"   50th percentile: {np.percentile(volumes, 50):.1f} dB")
        print(f"   75th percentile: {np.percentile(volumes, 75):.1f} dB")
        print(f"   90th percentile: {np.percentile(volumes, 90):.1f} dB")
        
        # Recommend thresholds
        mean_vol = np.mean(volumes)
        suggested_thresholds = [
            mean_vol - 10,  # Conservative
            mean_vol - 5,   # Moderate  
            mean_vol,       # Mean-based
            mean_vol + 3,   # Aggressive
            mean_vol + 5    # Very aggressive
        ]
        
        print(f"\n🎯 Suggested silence thresholds:")
        for i, thresh in enumerate(suggested_thresholds):
            labels = ["Conservative", "Moderate", "Mean-based", "Aggressive", "Very Aggressive"]
            print(f"   {labels[i]:15}: {thresh:.1f} dB")

def main():
    audio_file = "interview_cut.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print("🎙️ SILENCE DETECTION DIAGNOSIS")
    print("=" * 50)
    
    # Step 1: Test different silence detection parameters
    silence_results = analyze_silence_detection(audio_file)
    
    # Step 2: Analyze volume distribution
    analyze_audio_volume_distribution(audio_file)
    
    # Step 3: Generate recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    if silence_results:
        best = max(silence_results, key=lambda x: x["speech_percentage"])
        config = best["config"]
        
        print(f"✅ Use these parameters for better detection:")
        print(f"   min_silence_len: {config['min_silence_len']}")
        print(f"   silence_thresh: {config['silence_thresh']}")
        print(f"   Expected segments: {best['segments']}")
        print(f"   Speech coverage: {best['speech_percentage']:.1f}%")
        
        if best["first_speech"] > 10:
            print(f"⚠️  WARNING: First speech starts at {best['first_speech']:.1f}s - check if early content is missed!")
    
    print(f"\n🔧 Next steps:")
    print("1. Test with the recommended parameters")
    print("2. If still issues, try manual segmentation")
    print("3. Check if audio normalization helps")

if __name__ == "__main__":
    main()
