#!/usr/bin/env python3
"""
Study Material Processor v2.0 - Modular Version
A complete solution for processing lecture videos with transcription, 
screenshot extraction, and PDF linking using a clean modular architecture.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import only what's needed for basic functionality
from src.config import WHISPER_MODELS, LANGUAGE_MAP, LOGGING_CONFIG
from src.html_generator import HTMLReportGenerator

def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging based on verbosity level.
    
    Args:
        verbose: Enable verbose logging
        debug: Enable debug logging
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format=LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['datefmt']
    )

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Study Material Processor v2.0 - Process lecture videos with transcription and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python study_processor_v2.py --input video.mp4 --output ./results

  # Process all videos in directory
  python study_processor_v2.py --input ./videos --batch --output ./results --studies ./pdfs

  # Quick analysis with basic model
  python study_processor_v2.py --input video.mp4 --model base --no-screenshots

  # High quality processing
  python study_processor_v2.py --input video.mp4 --model large-v3 --similarity-threshold 0.80
  
  # Regenerate HTML/TXT reports from existing JSON files (fast!)
  python study_processor_v2.py --input mad --regenerate-reports
  python study_processor_v2.py --input "mad/Video.mp4.json" --regenerate-reports
        """
    )
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                       help="Video file or directory containing videos")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: same directory as input file)")
    parser.add_argument("--studies", type=str, default="./studies",
                       help="Directory with study materials/PDFs (default: ./studies)")
    
    # Processing modes
    parser.add_argument("--batch", action="store_true",
                       help="Process all videos in input directory")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze video complexity without processing")
    parser.add_argument("--regenerate-reports", action="store_true",
                       help="Regenerate HTML/TXT reports from existing JSON files (no transcription/screenshots)")
    
    # Transcription settings
    transcription_group = parser.add_argument_group("Transcription Settings")
    transcription_group.add_argument("--language", type=str, default="german",
                                   choices=list(LANGUAGE_MAP.keys()),
                                   help="Language for transcription (default: german)")
    transcription_group.add_argument("--model", type=str, default="large-v3",
                                   choices=WHISPER_MODELS,
                                   help="Whisper model (default: large-v3)")
    transcription_group.add_argument("--device", type=str, default=None,
                                   help="Device for inference (cpu, cuda, etc.)")
    transcription_group.add_argument("--no-segmentation", action="store_true",
                                   help="Disable audio segmentation (process entire file). Default: use segmentation for better quality")
    transcription_group.add_argument("--segmentation", action="store_true",
                                   help="Enable audio segmentation (default, kept for compatibility)")
    transcription_group.add_argument("--split-audio", action="store_true",
                                   help="Alias for --segmentation")
    
    # Screenshot settings
    screenshot_group = parser.add_argument_group("Screenshot Settings")
    screenshot_group.add_argument("--no-screenshots", action="store_true",
                                help="Disable screenshot extraction")
    screenshot_group.add_argument("--similarity-threshold", type=float, default=0.85,
                                help="Scene change detection threshold (default: 0.85)")
    screenshot_group.add_argument("--min-interval", type=float, default=2.0,
                                help="Minimum seconds between screenshots (default: 2.0)")
    
    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument("--no-html", action="store_true",
                            help="Disable HTML report generation")
    output_group.add_argument("--no-json", action="store_true",
                            help="Disable JSON output")
    output_group.add_argument("--cleanup-audio", action="store_true",
                            help="Remove extracted audio files after processing")
    
    # Advanced settings
    advanced_group = parser.add_argument_group("Advanced Settings")
    advanced_group.add_argument("--config", type=str,
                              help="Path to custom configuration file")
    advanced_group.add_argument("--validate", action="store_true",
                              help="Validate setup and exit")
    
    # Logging
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--verbose", action="store_true",
                             help="Enable verbose output")
    logging_group.add_argument("--debug", action="store_true",
                             help="Enable debug output")
    logging_group.add_argument("--quiet", action="store_true",
                             help="Suppress all output except errors")
    
    return parser

def load_custom_config(config_path: str) -> Dict:
    """
    Load custom configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)

def create_config_from_args(args) -> Dict:
    """
    Create configuration dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        'transcription': {
            'model': args.model,
            'language': args.language,
            'device': args.device,
            'disable_segmentation': args.no_segmentation,  # Default: segmentation enabled for better quality
        },
        'screenshots': {
            'similarity_threshold': args.similarity_threshold,
            'min_time_between_shots': args.min_interval,
        },
        'output': {
            'extract_screenshots': not args.no_screenshots,
            'generate_html': not args.no_html,
            'generate_json': not args.no_json,
            'cleanup_audio': args.cleanup_audio,
        }
    }
    
    return config

def validate_inputs(args) -> None:
    """
    Validate command line inputs.
    
    Args:
        args: Parsed command line arguments
    """
    # Skip validation for regenerate mode (handled separately)
    if args.regenerate_reports:
        return
    
    # Check input path
    if not os.path.exists(args.input):
        logging.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Check if input is file vs directory
    if os.path.isfile(args.input) and args.batch:
        logging.error("Cannot use --batch with a single file input")
        sys.exit(1)
    
    if os.path.isdir(args.input) and not args.batch:
        logging.error("Input is a directory but --batch not specified")
        sys.exit(1)
    
    # Check studies directory if provided
    if args.studies and not os.path.exists(args.studies):
        logging.warning(f"Studies directory does not exist: {args.studies}")


def regenerate_reports(input_path: str) -> None:
    """
    Regenerate HTML and TXT reports from existing JSON files.
    
    Args:
        input_path: Path to JSON file or directory containing JSON files
    """
    logger = logging.getLogger(__name__)
    html_generator = HTMLReportGenerator()
    
    input_path_obj = Path(input_path)
    json_files = []
    
    # Find JSON files
    if input_path_obj.is_file():
        if input_path_obj.suffix == '.json':
            json_files = [input_path_obj]
        else:
            logger.error(f"Input file is not a JSON file: {input_path}")
            sys.exit(1)
    elif input_path_obj.is_dir():
        # Find all .mp4.json files in directory
        json_files = list(input_path_obj.glob("*.mp4.json"))
        if not json_files:
            logger.error(f"No JSON files found in directory: {input_path}")
            sys.exit(1)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    print(f"\n🔄 Regenerating reports from {len(json_files)} JSON file(s)...")
    print(f"   Input: {input_path}")
    print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    success_count = 0
    
    for json_file in json_files:
        base_name = json_file.stem  # e.g., "Video Name.mp4"
        print(f"📄 Processing: {base_name}")
        
        # Load JSON data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        except Exception as e:
            print(f"   ❌ Failed to load JSON: {e}")
            continue
        
        # Generate HTML report
        html_dest = json_file.parent / f"{base_name}.html"
        try:
            html_generator.generate_report(analysis_data, str(html_dest))
            print(f"   ✅ Generated HTML: {html_dest.name}")
        except Exception as e:
            print(f"   ❌ Failed to generate HTML: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generate TXT transcript
        txt_dest = json_file.parent / f"{base_name}.txt"
        try:
            # Handle nested transcription structure
            transcription_data = analysis_data.get('transcription', {})
            if isinstance(transcription_data, dict) and 'transcription' in transcription_data:
                actual_transcription = transcription_data['transcription']
            else:
                actual_transcription = transcription_data
            
            # Extract plain text
            segments = actual_transcription.get('segments', [])
            if segments:
                lines = []
                lines.append("=" * 80)
                lines.append("TRANSCRIPTION")
                lines.append("=" * 80)
                lines.append("")
                
                # Add metadata
                if actual_transcription.get('language'):
                    lines.append(f"Language: {actual_transcription['language']}")
                
                duration = segments[-1].get('end', 0) if segments else 0
                lines.append(f"Duration: {duration / 60:.1f} minutes ({duration:.1f} seconds)")
                lines.append(f"Segments: {len(segments)}")
                
                word_count = sum(len(seg.get('text', '').split()) for seg in segments)
                lines.append(f"Words: {word_count}")
                lines.append("")
                lines.append("=" * 80)
                lines.append("")
                
                # Add segments with timestamps
                for segment in segments:
                    start = segment.get('start', 0)
                    text = segment.get('text', '').strip()
                    
                    if not text:
                        continue
                    
                    # Format timestamp
                    start_h = int(start // 3600)
                    start_m = int((start % 3600) // 60)
                    start_s = int(start % 60)
                    
                    timestamp = f"[{start_h:02d}:{start_m:02d}:{start_s:02d}]"
                    lines.append(f"{timestamp} {text}")
                    lines.append("")
                
                txt_content = "\n".join(lines)
                
                with open(txt_dest, 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                print(f"   ✅ Generated TXT: {txt_dest.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to generate TXT: {e}")
        
        success_count += 1
        print()
    
    print(f"✅ Report regeneration completed!")
    print(f"   Successfully regenerated: {success_count}/{len(json_files)} files")
    print(f"   Output directory: {os.path.abspath(input_path_obj if input_path_obj.is_dir() else input_path_obj.parent)}")
    print(f"   Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main entry point."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        setup_logging(False, False)
        logging.getLogger().setLevel(logging.ERROR)
    else:
        setup_logging(args.verbose, args.debug)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Handle regenerate-reports mode first (doesn't need full setup)
        if args.regenerate_reports:
            regenerate_reports(args.input)
            return
        
        # Import heavy dependencies only when needed for processing
        from src.processor import StudyMaterialProcessor
        from src.utils import check_dependencies
        
        # Validate dependencies first
        logger.info("Checking dependencies...")
        check_dependencies()
        logger.info("All dependencies are available")
        
        # Validate inputs
        validate_inputs(args)
        
        # Load configuration
        if args.config:
            config = load_custom_config(args.config)
            logger.info(f"Loaded custom configuration from: {args.config}")
        else:
            config = create_config_from_args(args)
        
        # Initialize processor
        logger.info("Initializing Study Material Processor...")
        processor = StudyMaterialProcessor(config)
        
        # Validate setup if requested
        if args.validate:
            logger.info("Validating setup...")
            validation_results = processor.validate_setup()
            
            if validation_results["valid"]:
                print("✅ Setup validation successful!")
                if validation_results["warnings"]:
                    print("⚠️  Warnings:")
                    for warning in validation_results["warnings"]:
                        print(f"   - {warning}")
            else:
                print("❌ Setup validation failed!")
                for error in validation_results["errors"]:
                    print(f"   - {error}")
                sys.exit(1)
            
            return
        
        # Analysis mode
        if args.analyze_only:
            if os.path.isdir(args.input):
                logger.error("Analysis mode only supports single video files")
                sys.exit(1)
            
            logger.info(f"Analyzing video complexity: {args.input}")
            analysis = processor.analyze_video_complexity(args.input)
            
            print(f"\n📊 Video Analysis Results for: {Path(args.input).name}")
            print(f"Duration: {analysis['video_info']['duration_seconds']/60:.1f} minutes")
            print(f"Resolution: {analysis['video_info']['width']}x{analysis['video_info']['height']}")
            print(f"Complexity: {analysis['screenshot_analysis'].get('complexity', 'unknown')}")
            print(f"Recommended model: {analysis['recommended_settings']['model']}")
            print(f"Recommended screenshot threshold: {analysis['recommended_settings']['screenshot_threshold']}")
            print(f"Estimated processing time: {analysis['time_estimates']['estimated_total_minutes']:.1f} minutes")
            
            return
        
        # Create output directory if specified
        if args.output:
            os.makedirs(args.output, exist_ok=True)
        
        # Track total processing time
        total_start_time = time.time()
        
        # Process videos
        if args.batch:
            logger.info(f"Starting batch processing of directory: {args.input}")
            print(f"\n🚀 Starting batch processing...")
            print(f"   Mode: {'Whole-File (no segmentation)' if args.no_segmentation else 'Segmented (recommended)'}")
            print(f"   Input: {args.input}")
            print(f"   Output: {args.output if args.output else 'Same as input (source directory)'}")
            print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            results = processor.process_batch(args.input, args.output, args.studies)
            
            total_time = time.time() - total_start_time
            
            print(f"\n✅ Batch processing completed!")
            print(f"   Processed: {len(results)} videos")
            print(f"   Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
            print(f"   Average per video: {total_time/len(results)/60:.2f} minutes" if results else "")
            print(f"   Output directory: {os.path.abspath(args.output)}")
            print(f"   Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if len(results) > 1:
                index_path = os.path.join(args.output, "index.html")
                if os.path.exists(index_path):
                    print(f"   Index page: {index_path}")
            
        else:
            logger.info(f"Processing single video: {args.input}")
            print(f"\n🚀 Starting video processing...")
            print(f"   Mode: {'Whole-File (no segmentation)' if args.no_segmentation else 'Segmented (recommended)'}")
            print(f"   Input: {args.input}")
            print(f"   Output: {args.output if args.output else 'Same as input (source directory)'}")
            print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            result = processor.process_video(args.input, args.output, args.studies)
            
            video_name = Path(args.input).stem
            # Determine actual output directory from result
            output_dir = result.get('output_directory', args.output if args.output else Path(args.input).parent)
            
            print(f"\n✅ Processing completed!")
            print(f"   Video: {Path(args.input).name}")
            # Correctly access duration from the new structure and convert seconds to minutes
            duration_seconds = result.get('transcription', {}).get('speech_pattern_analysis', {}).get('duration_seconds')
            if duration_seconds is not None:
                print(f"   Duration: {duration_seconds / 60:.1f} minutes")
            else:
                print("   Duration: Not available")
            print(f"   Screenshots: {len(result['screenshots'])}")
            print(f"   Related PDFs: {len(result['related_pdfs'])}")
            print(f"   Processing time: {result['processing_time_seconds']:.1f} seconds")
            print(f"   Output directory: {os.path.abspath(output_dir)}")
            
            # Show specific output files
            html_report = os.path.join(output_dir, f"{video_name}_report.html")
            if os.path.exists(html_report):
                print(f"   📄 HTML Report: {html_report}")
            
            json_file = os.path.join(output_dir, f"{video_name}_analysis.json")
            if os.path.exists(json_file):
                print(f"   📊 JSON Data: {json_file}")
            
            txt_file = os.path.join(output_dir, f"{video_name}_transcript.txt")
            if os.path.exists(txt_file):
                print(f"   📝 Plain Text: {txt_file}")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n⏹️  Processing interrupted")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()