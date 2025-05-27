#!/usr/bin/env python3
"""
Test script for the modular v2.0 system with a single video
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test the v2.0 system with a single video."""
    
    script_dir = Path(__file__).parent
    studies_dir = script_dir / "studies"
    output_dir = script_dir / "test_output_v2"
    
    # Find first video file
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
    
    print(f"🎬 Testing with video: {video_file.name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check video info first
    analyze_cmd = [
        sys.executable, "study_processor_v2.py",
        "--input", str(video_file),
        "--analyze-only",
        "--verbose"
    ]
    
    print(f"\n🔍 First, let's analyze the video complexity:")
    print(f"   {' '.join(analyze_cmd)}")
    
    try:
        subprocess.run(analyze_cmd, cwd=script_dir, check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Analysis failed, but continuing with processing...")
    
    # Confirm processing
    response = input(f"\n🚀 Proceed with processing {video_file.name}? (y/N): ")
    if response.lower() != 'y':
        print("⏹️  Test cancelled.")
        sys.exit(0)
    
    # Create test output directory
    output_dir.mkdir(exist_ok=True)
    
    # Run with test settings (faster model, less strict screenshot detection)
    cmd = [
        sys.executable, "study_processor_v2.py",
        "--input", str(video_file),
        "--output", str(output_dir),
        "--studies", str(studies_dir),
        "--model", "base",  # Faster model for testing
        "--similarity-threshold", "0.80",  # More sensitive screenshot detection
        "--min-interval", "3.0",  # More screenshots for testing
        "--cleanup-audio",
        "--verbose"
    ]
    
    print(f"\n🔧 Running test with command:")
    print(f"   {' '.join(cmd)}")
    print(f"\n⚙️  Using 'base' model for faster testing")
    print(f"💡 For production, use 'large-v3' model for better quality")
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print(f"\n✅ Test completed successfully!")
        
        # Show output structure
        video_name = video_file.stem
        video_output_dir = output_dir / video_name
        
        if video_output_dir.exists():
            print(f"\n📁 Generated files in {video_output_dir}:")
            for file in video_output_dir.rglob("*"):
                if file.is_file():
                    relative_path = file.relative_to(output_dir)
                    file_size = file.stat().st_size
                    size_str = f"{file_size/1024:.1f} KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f} MB"
                    print(f"   📄 {relative_path} ({size_str})")
            
            # Highlight main files
            html_report = video_output_dir / f"{video_name}_report.html"
            json_data = video_output_dir / f"{video_name}_analysis.json"
            screenshots_dir = video_output_dir / "screenshots"
            
            print(f"\n🎯 Key outputs:")
            if html_report.exists():
                print(f"   🌐 HTML Report: {html_report}")
            if json_data.exists():
                print(f"   📊 JSON Data: {json_data}")
            if screenshots_dir.exists():
                screenshot_count = len(list(screenshots_dir.glob("*.jpg")))
                print(f"   📸 Screenshots: {screenshot_count} images in {screenshots_dir}")
        
        print(f"\n💡 To process all videos, run: python process_studies_v2.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error code {e.returncode}")
        print("🔧 Troubleshooting tips:")
        print("   1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("   2. Ensure ffmpeg is available in PATH")
        print("   3. Run with --debug for detailed error information:")
        debug_cmd = cmd + ["--debug"]
        print(f"      {' '.join(debug_cmd)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()