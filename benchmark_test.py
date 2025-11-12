#!/usr/bin/env python3
"""
Quick benchmark comparison script.
Test multiple models and compare their performance.
"""

import sys
import subprocess
from pathlib import Path

def run_benchmark_test(video_file: str):
    """
    Run benchmark tests with different models and compare results.
    """
    if not Path(video_file).exists():
        print(f"❌ Error: Video file not found: {video_file}")
        return
    
    print("="*80)
    print("🧪 BENCHMARK TEST - Multiple Models")
    print("="*80)
    print(f"📹 Test File: {video_file}\n")
    
    models = ['large-v3', 'large-v3-turbo', 'medium']
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"🔬 Test {i}/{len(models)}: Model '{model}'")
        print("="*80)
        
        try:
            # Run processing
            cmd = [
                'python', 'study_processor_v2.py',
                '--input', video_file,
                '--model', model,
                '--no-segmentation'  # For faster testing
            ]
            
            print(f"▶️  Running: {' '.join(cmd)}\n")
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to process with model {model}: {e}")
            continue
        except KeyboardInterrupt:
            print("\n⚠️  Benchmark test interrupted by user")
            sys.exit(1)
    
    # Show results
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS")
    print("="*80 + "\n")
    
    subprocess.run(['python', 'view_benchmarks.py', '--last', str(len(models))])
    
    print("\n" + "="*80)
    print("💡 TIP: Run 'python view_benchmarks.py --model MODEL_NAME' for detailed stats")
    print("="*80 + "\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python benchmark_test.py <video_file>")
        print("\nExample:")
        print("  python benchmark_test.py test_video.mp4")
        sys.exit(1)
    
    run_benchmark_test(sys.argv[1])
