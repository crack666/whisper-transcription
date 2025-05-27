#!/usr/bin/env python3
"""
Test script specifically for slow speakers with long pauses (like professors)
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test processing with optimized settings for slow speakers."""
    
    script_dir = Path(__file__).parent
    studies_dir = script_dir / "studies"
    output_dir = script_dir / "test_slow_speaker"
    
    # Find video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_file = None
    
    for ext in video_extensions:
        videos = list(studies_dir.glob(f"*{ext}"))
        if videos:
            video_file = videos[0]
            break
    
    if not video_file:
        print("❌ No video files found in studies directory")
        sys.exit(1)
    
    print(f"🎓 Testing Slow Speaker Optimization")
    print(f"Video: {video_file.name}")
    print(f"Output: {output_dir}")
    
    # First, analyze the audio
    print(f"\n📊 Step 1: Analyzing audio characteristics...")
    
    analyze_cmd = [
        sys.executable, "transcription_analyzer.py",
        "--audio", str(video_file),
        "--output", str(output_dir / "analysis"),
        "--compare",
        "--visualize",
        "--model", "base"  # Quick analysis
    ]
    
    try:
        subprocess.run(analyze_cmd, cwd=script_dir, check=True)
        print("✅ Analysis completed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Analysis failed ({e.returncode}), continuing with processing...")
    except Exception as e:
        print(f"⚠️ Analysis error: {e}")
    
    # Ask user about processing
    response = input(f"\n🚀 Proceed with optimized slow speaker processing? (y/N): ")
    if response.lower() != 'y':
        print("⏹️ Test cancelled.")
        sys.exit(0)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Test with different configurations
    configs_to_test = [
        ("standard", None, "Standard transcriber"),
        ("slow_speaker", "configs/slow_speaker.json", "Slow speaker optimized"),
        ("lecture_optimized", "configs/lecture_optimized.json", "Lecture optimized (max settings)")
    ]
    
    results = {}
    
    for config_name, config_file, description in configs_to_test:
        print(f"\n🧪 Testing: {description}")
        
        test_output = output_dir / config_name
        test_output.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, "study_processor_v2.py",
            "--input", str(video_file),
            "--output", str(test_output),
            "--studies", str(studies_dir),
            "--model", "base",  # Use fast model for testing
            "--cleanup-audio",
            "--verbose"
        ]
        
        if config_file:
            cmd.extend(["--config", config_file])
        else:
            # Standard settings - disable enhanced transcriber
            cmd.extend([
                "--similarity-threshold", "0.85",
                "--min-interval", "2.0"
            ])
        
        try:
            print(f"   Running: {' '.join(cmd[-6:])}")  # Show last 6 args
            result = subprocess.run(cmd, cwd=script_dir, check=True, 
                                  capture_output=True, text=True)
            
            # Try to extract results
            video_name = video_file.stem
            json_file = test_output / video_name / f"{video_name}_analysis.json"
            
            if json_file.exists():
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                transcription = data.get('transcription', {})
                results[config_name] = {
                    "description": description,
                    "word_count": len(transcription.get('full_text', '').split()),
                    "segments": len(transcription.get('segments', [])),
                    "duration": transcription.get('total_duration', 0) / 1000 / 60,
                    "success_rate": (transcription.get('segments_successful', 0) / 
                                   max(transcription.get('segments_total', 1), 1)),
                    "processing_time": data.get('processing_time_seconds', 0),
                    "quality_score": transcription.get('average_quality_score', 0)
                }
            else:
                results[config_name] = {
                    "description": description,
                    "error": "No results file found"
                }
            
            print(f"   ✅ Completed: {config_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e.returncode}")
            results[config_name] = {
                "description": description,
                "error": f"Process failed with code {e.returncode}"
            }
            continue
    
    # Compare results
    print(f"\n📊 Results Comparison:")
    print(f"{'Config':<20} {'Words':<8} {'Segments':<10} {'Success%':<10} {'Quality':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for config_name, result in results.items():
        if "error" in result:
            print(f"{config_name:<20} ERROR: {result['error']}")
        else:
            print(f"{result['description']:<20} "
                  f"{result['word_count']:<8} "
                  f"{result['segments']:<10} "
                  f"{result['success_rate']*100:<10.1f} "
                  f"{result.get('quality_score', 0):<10.2f} "
                  f"{result['processing_time']:<10.1f}")
    
    # Find best result
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if valid_results:
        best_config = max(valid_results.items(), 
                         key=lambda x: x[1]['word_count'] * x[1]['success_rate'])
        
        print(f"\n🏆 Best Result: {best_config[1]['description']}")
        print(f"   Words: {best_config[1]['word_count']}")
        print(f"   Success Rate: {best_config[1]['success_rate']*100:.1f}%")
        
        if best_config[1].get('quality_score', 0) > 0:
            print(f"   Quality Score: {best_config[1]['quality_score']:.2f}")
        
        # Show output location
        best_output = output_dir / best_config[0] / video_file.stem
        html_report = best_output / f"{video_file.stem}_report.html"
        
        if html_report.exists():
            print(f"   📄 Best Report: {html_report}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if "lecture_optimized" in valid_results and "standard" in valid_results:
        improvement = (valid_results["lecture_optimized"]["word_count"] - 
                      valid_results["standard"]["word_count"])
        if improvement > 50:
            print(f"   ✅ Use 'lecture_optimized' config (+{improvement} words vs standard)")
        elif improvement > 10:
            print(f"   ⚡ Consider 'slow_speaker' config for good balance")
        else:
            print(f"   ℹ️ Standard transcription seems adequate for this audio")
    
    print(f"\n📁 All results saved in: {output_dir}")
    print(f"📈 Analysis charts (if created): {output_dir / 'analysis'}")

if __name__ == "__main__":
    main()