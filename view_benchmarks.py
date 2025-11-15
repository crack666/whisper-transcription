"""
Benchmark analysis and reporting tool.
View performance statistics across multiple runs.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmark_logger import BenchmarkLogger

def main():
    parser = argparse.ArgumentParser(
        description="View benchmark statistics and performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all benchmark statistics
  python view_benchmarks.py
  
  # Filter by model
  python view_benchmarks.py --model large-v3
  
  # Filter by segmentation mode
  python view_benchmarks.py --segmentation-mode defensive_silence
  
  # View raw benchmark data
  python view_benchmarks.py --raw
  
  # Export to JSON
  python view_benchmarks.py --export stats.json
        """
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./benchmark_logs',
        help='Directory containing benchmark logs (default: ./benchmark_logs)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Filter by model name (e.g., large-v3, medium)'
    )
    
    parser.add_argument(
        '--segmentation-mode',
        type=str,
        help='Filter by segmentation mode (e.g., defensive_silence, adaptive)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help='Filter by device (e.g., cuda, cpu)'
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Show raw benchmark data instead of statistics'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export statistics to JSON file'
    )
    
    parser.add_argument(
        '--last',
        type=int,
        help='Show only last N runs'
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = BenchmarkLogger(log_dir=args.log_dir)
    
    # Build filter
    filter_by = {}
    if args.model:
        filter_by['config.model'] = args.model
    if args.segmentation_mode:
        filter_by['config.segmentation_mode'] = args.segmentation_mode
    if args.device:
        filter_by['config.device'] = args.device
    
    # Get statistics
    stats = logger.get_statistics(filter_by if filter_by else None)
    
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return
    
    if args.raw:
        # Show raw data
        import json
        if args.last:
            stats['recent_runs'] = stats['recent_runs'][-args.last:]
        print(json.dumps(stats['recent_runs'], indent=2))
    elif args.export:
        # Export to file
        import json
        with open(args.export, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Statistics exported to {args.export}")
    else:
        # Print formatted report
        logger.print_report(filter_by if filter_by else None)
        
        # Show recent runs summary
        if stats.get('recent_runs'):
            print("\n📋 Recent Runs:")
            print("-" * 80)
            
            recent = stats['recent_runs'][-10:] if not args.last else stats['recent_runs'][-args.last:]
            
            for run in recent:
                status = "✅" if run.get('success') else "❌"
                model = run.get('config', {}).get('model', 'unknown')
                processing_mode = run.get('config', {}).get('processing_mode', 'Unknown')
                duration = run.get('file', {}).get('duration_seconds', 0)
                processing_time = run.get('total_duration_seconds', 0)
                speedup = run.get('metrics', {}).get('speedup', 0)
                timestamp = run.get('timestamp', 'unknown')[:19]  # Remove microseconds
                gpu_name = run.get('hardware', {}).get('gpu', {}).get('name', 'CPU')
                if gpu_name != 'CPU':
                    gpu_name = gpu_name.split()[-1] if gpu_name else 'GPU'  # Shorten GPU name
                
                print(f"{status} [{timestamp}] {model:15s} {processing_mode:10s} {gpu_name:8s} | "
                      f"{duration:6.1f}s media → {processing_time:6.1f}s process | "
                      f"{speedup:5.2f}x speedup")

if __name__ == '__main__':
    main()
