#!/usr/bin/env python3
"""
Batch Timeline Report Generator

Generates timeline reports for all analysis results in the results directory.
This is a simplified version focused specifically on creating timeline reports.

Usage:
    python batch_generate_timeline_reports.py [options]
    
Options:
    --verbose       Enable verbose logging
    --regenerate    Regenerate even if timeline report already exists
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from html_generator import HTMLReportGenerator
from utils import sanitize_filename

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def find_all_analysis_files() -> List[Path]:
    """Find all analysis JSON files"""
    results_dir = Path("results")
    analysis_files = []
    
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            analysis_pattern = f"{subdir.name}_analysis.json"
            analysis_file = subdir / analysis_pattern
            
            if analysis_file.exists():
                analysis_files.append(analysis_file)
                
    return analysis_files

def generate_timeline_report_for_file(analysis_file: Path, regenerate: bool = False, logger=None) -> bool:
    """Generate timeline report for a single analysis file"""
    try:
        # Load analysis data
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
            
        # Get output directory and video name
        output_dir = Path(analysis_data.get("output_directory", analysis_file.parent))
        video_name = sanitize_filename(Path(analysis_data.get("video_path", "unknown")).stem)
        
        # Check if timeline report already exists
        timeline_report_path = output_dir / f"{video_name}_report_TIMELINE.html"
        
        if timeline_report_path.exists() and not regenerate:
            logger.info(f"⏭️  Timeline report already exists: {timeline_report_path.name}")
            return True
            
        logger.info(f"🕐 Generating timeline report: {timeline_report_path.name}")
        
        # Generate timeline report
        html_generator = HTMLReportGenerator()
        html_generator.generate_report(analysis_data, str(timeline_report_path))
        
        logger.info(f"   ✅ Timeline report generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed to generate timeline report for {analysis_file.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate timeline reports for all study material analysis results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--regenerate", "-r",
        action="store_true",
        help="Regenerate even if timeline report already exists"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("🚀 Starting batch timeline report generation...")
    
    # Find all analysis files
    analysis_files = find_all_analysis_files()
    
    if not analysis_files:
        logger.error("No analysis files found in results directory")
        return
        
    logger.info(f"📊 Found {len(analysis_files)} analysis files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, analysis_file in enumerate(analysis_files, 1):
        logger.info(f"\n[{i}/{len(analysis_files)}] Processing: {analysis_file.name}")
        
        if generate_timeline_report_for_file(analysis_file, args.regenerate, logger):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("🎉 BATCH TIMELINE GENERATION COMPLETED!")
    logger.info("="*60)
    logger.info(f"✅ Successfully generated: {successful}")
    logger.info(f"❌ Failed to generate: {failed}")
    logger.info(f"📊 Total processed: {len(analysis_files)}")
    
    if successful > 0:
        logger.info(f"\n🌐 Update master index to see all timeline reports:")
        logger.info(f"   python generate_master_index.py")
        
    logger.info("="*60)

if __name__ == "__main__":
    main()
