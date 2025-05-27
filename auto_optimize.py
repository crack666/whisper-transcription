#!/usr/bin/env python3
"""
Auto-Optimization Script for Whisper Transcription.

This script automatically finds the optimal transcription settings for any audio file
by analyzing its characteristics and learning from previous optimizations.

Usage:
    python auto_optimize.py --input video.mp4 [--quick] [--output config.json]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive_optimizer import AdaptiveOptimizer

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Auto-optimize transcription settings for audio/video files"
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input video/audio file to optimize for"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output path for optimized configuration file (default: auto-generated)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick optimization with fewer test configurations"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging output"
    )
    
    parser.add_argument(
        "--max-configs",
        type=int,
        default=8,
        help="Maximum number of configurations to test (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"❌ Input file not found: {args.input}")
        return 1
    
    # Determine output path
    if not args.output:
        input_name = Path(args.input).stem
        timestamp = int(time.time())
        args.output = f"configs/auto_optimized_{input_name}_{timestamp}.json"
    
    # Ensure configs directory exists
    os.makedirs("configs", exist_ok=True)
    
    # Quick mode uses fewer configurations
    max_configs = 4 if args.quick else args.max_configs
    
    print(f"🎯 Auto-Optimization for Whisper Transcription")
    print(f"=" * 50)
    print(f"📁 Input file: {args.input}")
    print(f"💾 Output config: {args.output}")
    print(f"⚡ Mode: {'Quick' if args.quick else 'Comprehensive'}")
    print(f"🧪 Max configs to test: {max_configs}")
    print()
    
    try:
        # Initialize adaptive optimizer
        optimizer = AdaptiveOptimizer()
        
        # Run adaptive optimization
        logger.info("🚀 Starting adaptive optimization...")
        analysis = optimizer.run_adaptive_optimization(args.input, max_configs)
        
        # Check if optimization was successful
        if "best_config" in analysis and analysis["successful_tests"] > 0:
            # Find the best result to create config file
            best_result = None
            for result in optimizer.optimization_history:
                if (result.config_name == analysis["best_config"]["name"] and 
                    result.word_count == analysis["best_config"]["word_count"]):
                    best_result = result
                    break
            
            if best_result:
                # Create optimized configuration file
                optimizer.create_optimized_config_file(best_result, args.output)
                
                # Print summary
                print(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
                print(f"=" * 50)
                print(f"🏆 Best Configuration: {best_result.config_name}")
                print(f"📝 Word Count: {best_result.word_count}")
                print(f"⏱️  Processing Time: {best_result.processing_time:.1f}s")
                print(f"🎭 Speaker Type: {best_result.audio_profile.speaker_type}")
                print(f"🗣️  Speech Ratio: {best_result.audio_profile.speech_ratio:.2f}")
                print(f"💾 Config saved to: {args.output}")
                
                # Show key parameters
                params = best_result.parameters
                print(f"\n🔧 Optimal Parameters:")
                print(f"   Min Silence Length: {params.get('min_silence_len', 'N/A')}ms")
                print(f"   Silence Adjustment: {params.get('silence_adjustment', 'N/A')}dB")
                print(f"   Padding: {params.get('padding', 'N/A')}ms")
                
                # Usage instructions
                print(f"\n📖 Usage Instructions:")
                print(f"   Use this optimized config with:")
                print(f"   python study_processor_v2.py --config {args.output} --input your_video.mp4")
                
                return 0
            else:
                logger.error("❌ Could not find best result in optimization history")
                return 1
        else:
            logger.error("❌ Optimization failed - no successful configurations found")
            print(f"\n💥 OPTIMIZATION FAILED")
            print(f"   Total tests: {analysis.get('total_tests', 0)}")
            print(f"   Successful: {analysis.get('successful_tests', 0)}")
            
            if "error" in analysis:
                print(f"   Error: {analysis['error']}")
            
            return 1
    
    except Exception as e:
        logger.error(f"❌ Optimization error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())