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
    
    # Suppress excessive debug output from NumPy, Numba, and other libraries
    # when verbose mode is enabled, keeping only application-level debug info
    if verbose:
        # Set specific loggers to WARNING to reduce noise
        noisy_loggers = [
            'numba.core.ssa',
            'numba.core.interpreter', 
            'numba.core.byteflow',
            'numba.core.ir_utils',
            'numba.core.transforms',
            'numba.core.analysis',
            'numba.core.typed_passes',
            'numba.core.untyped_passes',
            'numba.core.pipeline',
            'numba.core.compiler_machinery',
            'numba.parfors',
            'numba',
            'numpy',
            'matplotlib',
            'PIL'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Auto-optimize transcription settings for audio/video files"
    )
    
    parser.add_argument(
        "--input", "-i", 
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
    
    parser.add_argument(
        "--sequential",
        action="store_true", 
        help="Use sequential processing instead of parallel (safer for memory)"
    )
    
    parser.add_argument(
        "--test-file",
        action="store_true",
        help="Use short test file (TestFile_cut.mp4) instead of specified input"
    )
    
    parser.add_argument(
        "--analyze-params",
        action="store_true",
        help="Analyze parameter impact from optimization history without running new tests"
    )
    
    args = parser.parse_args()
    
    # Use test file if specified or if no input given
    if args.test_file or not args.input:
        test_path = "TestFile_cut.mp4"
        if os.path.exists(test_path):
            args.input = test_path
            print(f"🧪 Using test file: {test_path}")
        else:
            if not args.input:
                print(f"❌ No input file specified and test file not found: {test_path}")
                return 1
            print(f"⚠️ Test file not found: {test_path}, using specified input")
    
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
    print(f"🔄 Processing: {'Sequential' if args.sequential else 'Parallel (shared model)'}")
    print()
    
    try:
        # Initialize adaptive optimizer
        optimizer = AdaptiveOptimizer()
        
        # Handle parameter analysis mode
        if args.analyze_params:
            print("📊 Analyzing parameter impact from optimization history...")
            print()
            
            analysis = optimizer.analyze_parameter_impact()
            
            if "error" in analysis:
                print(f"❌ {analysis['error']}")
                return 1
            
            # Display analysis results
            print("🎛️ Parameter Impact Analysis")
            print("=" * 50)
            print(f"📈 Total results analyzed: {analysis['total_results_analyzed']}")
            print()
            
            # Min silence length analysis
            silence_analysis = analysis["min_silence_len_analysis"]
            print("🔇 Min Silence Length Analysis:")
            print(f"   Best range: {silence_analysis['best_range']} (avg: {silence_analysis['best_range_avg_words']:.1f} words)")
            print(f"   Recommendation: {silence_analysis['recommendation']}")
            print()
            
            # Silence adjustment analysis
            adj_analysis = analysis["silence_adjustment_analysis"]
            print("🎚️ Silence Adjustment Analysis:")
            print(f"   Best range: {adj_analysis['best_range']} (avg: {adj_analysis['best_range_avg_words']:.1f} words)")
            print(f"   Recommendation: {adj_analysis['recommendation']}")
            print()
            
            # Top performing configs
            print("🏆 Top Performing Configurations:")
            for i, config in enumerate(analysis["top_performing_configs"][:3], 1):
                print(f"   {i}. {config['name']}: {config['word_count']} words ({config['efficiency']:.1f} w/s)")
                params = config['parameters']
                print(f"      silence_len: {params['min_silence_len']}ms, adjustment: {params['silence_adjustment']}")
            print()
            
            # Key insights
            print("💡 Optimization Insights:")
            for insight in analysis["optimization_insights"]:
                print(f"   • {insight}")
            
            # Save analysis to file
            analysis_file = f"parameter_analysis_{int(time.time())}.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Full analysis saved to: {analysis_file}")
            return 0
        
        # Validate input file for optimization mode
        if not args.input:
            print("❌ No input file specified. Use --input or --test-file flag.")
            return 1
        
        # Run adaptive optimization
        logger.info("🚀 Starting adaptive optimization...")
        analysis = optimizer.run_adaptive_optimization(
            args.input, 
            max_configs, 
            use_parallel=not args.sequential
        )
        
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