#!/usr/bin/env python3
"""
Mass Regeneration Script for Study Material Processor

This script regenerates screenshots and HTML reports for all existing analysis results,
and creates a comprehensive index page linking all reports together.

Features:
- Batch processing of all analysis.json files in results/
- Regenerates screenshots with improved extraction logic
- Creates both standard and timeline HTML reports
- Generates master index page with links to all reports
- Progress tracking and error handling
- Optional parameter overrides for screenshot generation

Usage:
    python regenerate_all_results.py [options]
    
Options:
    --similarity_threshold FLOAT    Override similarity threshold (0.0-1.0)
    --min_time_between_shots FLOAT  Override min time between screenshots (seconds)
    --skip_screenshots              Skip screenshot regeneration, only regenerate reports
    --timeline_only                 Only generate timeline reports
    --verbose                       Enable verbose logging
"""

import argparse
import json
import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from video_processor import VideoScreenshotExtractor
from html_generator import HTMLReportGenerator
from config import DEFAULT_CONFIG
from utils import sanitize_filename, ensure_directory

# Setup logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('regenerate_all_results.log', mode='w', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

class MassRegenerator:
    """Handles mass regeneration of screenshots and reports for all analysis results"""
    
    def __init__(self, similarity_threshold: Optional[float] = None, 
                 min_time_between_shots: Optional[float] = None,
                 skip_screenshots: bool = False,
                 timeline_only: bool = False,
                 verbose: bool = False):
        self.logger = setup_logging(verbose)
        self.similarity_threshold = similarity_threshold
        self.min_time_between_shots = min_time_between_shots
        self.skip_screenshots = skip_screenshots
        self.timeline_only = timeline_only
        self.results_dir = Path("results")
        self.html_generator = HTMLReportGenerator()
        
        # Statistics tracking
        self.stats = {
            'total_found': 0,
            'processed_successfully': 0,
            'failed_processing': 0,
            'screenshots_regenerated': 0,
            'reports_generated': 0,
            'timeline_reports_generated': 0,
            'errors': []
        }
        
    def find_all_analysis_files(self) -> List[Path]:
        """Find all *_analysis.json files in the results directory"""
        analysis_files = []
        
        if not self.results_dir.exists():
            self.logger.error(f"Results directory not found: {self.results_dir}")
            return analysis_files
            
        for subdir in self.results_dir.iterdir():
            if subdir.is_dir():
                # Look for analysis.json files
                analysis_pattern = f"{subdir.name}_analysis.json"
                analysis_file = subdir / analysis_pattern
                
                if analysis_file.exists():
                    analysis_files.append(analysis_file)
                    self.logger.debug(f"Found analysis file: {analysis_file}")
                else:
                    self.logger.warning(f"No analysis file found in {subdir}")
                    
        self.stats['total_found'] = len(analysis_files)
        self.logger.info(f"📊 Found {len(analysis_files)} analysis files to process")
        return analysis_files
        
    def regenerate_screenshots_for_file(self, analysis_file: Path) -> bool:
        """Regenerate screenshots for a single analysis file"""
        try:
            # Load analysis data
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                
            # Get video path
            video_path_str = analysis_data.get("video_path") or analysis_data.get("audio_file_path")
            if not video_path_str:
                self.logger.error(f"No video path found in {analysis_file}")
                return False
                
            video_path = Path(video_path_str)
            if not video_path.exists():
                self.logger.error(f"Video file not found: {video_path}")
                return False
                
            # Get transcription segments
            transcription_data = analysis_data.get("transcription", {})
            if "transcription" in transcription_data:
                speech_segments = transcription_data["transcription"].get("segments", [])
            else:
                speech_segments = transcription_data.get("segments", [])
                
            # Get output directory
            if "output_directory" not in analysis_data:
                self.logger.error(f"No output_directory in {analysis_file}")
                return False
                
            output_dir = Path(analysis_data["output_directory"])
            screenshots_dir = ensure_directory(output_dir / "screenshots")
            
            # Get screenshot config
            original_config = analysis_data.get("config_used", {})
            screenshot_config = original_config.get("screenshots", DEFAULT_CONFIG['screenshots'].copy())
            
            # Apply overrides
            if self.similarity_threshold is not None:
                screenshot_config['similarity_threshold'] = self.similarity_threshold
            if self.min_time_between_shots is not None:
                screenshot_config['min_time_between_shots'] = self.min_time_between_shots
                
            self.logger.info(f"🖼️  Regenerating screenshots for: {video_path.name}")
            self.logger.info(f"   📁 Output: {screenshots_dir}")
            self.logger.info(f"   🎬 Segments: {len(speech_segments)}")
            
            # Create screenshot extractor
            extractor = VideoScreenshotExtractor(
                similarity_threshold=screenshot_config['similarity_threshold'],
                min_time_between_shots=screenshot_config['min_time_between_shots'],
                config=screenshot_config,
                speech_segments=speech_segments
            )
            
            # Extract screenshots
            new_screenshots = extractor.extract_screenshots(str(video_path), str(screenshots_dir))
            self.logger.info(f"   ✅ Generated {len(new_screenshots)} screenshots")
            
            # Update analysis data
            analysis_data["screenshots"] = new_screenshots
            analysis_data["processing_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save updated analysis data
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
                
            self.stats['screenshots_regenerated'] += len(new_screenshots)
            return True
            
        except Exception as e:
            self.logger.error(f"Error regenerating screenshots for {analysis_file}: {e}")
            self.stats['errors'].append(f"Screenshots {analysis_file.name}: {str(e)}")
            return False
            
    def generate_reports_for_file(self, analysis_file: Path) -> Tuple[bool, bool]:
        """Generate both standard and timeline reports for an analysis file"""
        standard_success = False
        timeline_success = False
        
        try:
            # Load analysis data
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                
            output_dir = Path(analysis_data.get("output_directory", analysis_file.parent))
            video_name = sanitize_filename(Path(analysis_data.get("video_path", "unknown")).stem)
            
            if not self.timeline_only:
                # Generate standard report
                standard_report_path = output_dir / f"{video_name}_report.html"
                self.logger.info(f"📄 Generating standard report: {standard_report_path.name}")
                
                try:
                    self.html_generator.generate_report(analysis_data, str(standard_report_path))
                    self.logger.info(f"   ✅ Standard report generated successfully")
                    standard_success = True
                    self.stats['reports_generated'] += 1
                except Exception as e:
                    self.logger.error(f"   ❌ Failed to generate standard report: {e}")
                    self.stats['errors'].append(f"Standard report {analysis_file.name}: {str(e)}")
            
            # Generate timeline report
            timeline_report_path = output_dir / f"{video_name}_report_TIMELINE.html"
            self.logger.info(f"🕐 Generating timeline report: {timeline_report_path.name}")
            
            try:
                self.html_generator.generate_report(analysis_data, str(timeline_report_path))
                self.logger.info(f"   ✅ Timeline report generated successfully")
                timeline_success = True
                self.stats['timeline_reports_generated'] += 1
            except Exception as e:
                self.logger.error(f"   ❌ Failed to generate timeline report: {e}")
                self.stats['errors'].append(f"Timeline report {analysis_file.name}: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error loading analysis file {analysis_file}: {e}")
            self.stats['errors'].append(f"Load {analysis_file.name}: {str(e)}")
            
        return standard_success, timeline_success
        
    def generate_master_index(self, all_analysis_data: List[Dict]) -> bool:
        """Generate master index page linking to all reports"""
        try:
            index_path = self.results_dir / "index.html"
            self.logger.info(f"📋 Generating master index page: {index_path}")
            
            # Filter out failed results
            successful_results = [data for data in all_analysis_data if not data.get("error")]
            
            self.html_generator.generate_index_page(successful_results, str(index_path))
            self.logger.info(f"   ✅ Master index generated with {len(successful_results)} entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate master index: {e}")
            self.stats['errors'].append(f"Master index: {str(e)}")
            return False
            
    def process_all(self) -> None:
        """Main processing function - regenerate everything"""
        start_time = time.time()
        self.logger.info("🚀 Starting mass regeneration of study material reports...")
        
        # Find all analysis files
        analysis_files = self.find_all_analysis_files()
        if not analysis_files:
            self.logger.error("No analysis files found. Nothing to process.")
            return
            
        all_analysis_data = []
        
        # Process each file
        for i, analysis_file in enumerate(analysis_files, 1):
            self.logger.info(f"\n📊 Processing {i}/{len(analysis_files)}: {analysis_file.name}")
            
            try:
                # Load analysis data for index generation
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                all_analysis_data.append(analysis_data)
                
                success = True
                
                # Regenerate screenshots if requested
                if not self.skip_screenshots:
                    screenshot_success = self.regenerate_screenshots_for_file(analysis_file)
                    if not screenshot_success:
                        success = False
                        
                # Generate reports
                standard_success, timeline_success = self.generate_reports_for_file(analysis_file)
                if not (standard_success or timeline_success):
                    success = False
                    
                if success:
                    self.stats['processed_successfully'] += 1
                    self.logger.info(f"   ✅ {analysis_file.name} processed successfully")
                else:
                    self.stats['failed_processing'] += 1
                    self.logger.warning(f"   ⚠️  {analysis_file.name} had some failures")
                    
            except Exception as e:
                self.logger.error(f"   ❌ Critical error processing {analysis_file.name}: {e}")
                self.stats['failed_processing'] += 1
                self.stats['errors'].append(f"Critical {analysis_file.name}: {str(e)}")
                
        # Generate master index
        self.logger.info("\n📋 Generating master index page...")
        index_success = self.generate_master_index(all_analysis_data)
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        self.print_final_statistics(elapsed_time, index_success)
        
    def print_final_statistics(self, elapsed_time: float, index_success: bool) -> None:
        """Print comprehensive processing statistics"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎉 MASS REGENERATION COMPLETED!")
        self.logger.info("="*80)
        self.logger.info(f"⏱️  Total processing time: {elapsed_time:.1f} seconds")
        self.logger.info(f"📁 Analysis files found: {self.stats['total_found']}")
        self.logger.info(f"✅ Successfully processed: {self.stats['processed_successfully']}")
        self.logger.info(f"❌ Failed processing: {self.stats['failed_processing']}")
        
        if not self.skip_screenshots:
            self.logger.info(f"🖼️  Screenshots regenerated: {self.stats['screenshots_regenerated']}")
            
        if not self.timeline_only:
            self.logger.info(f"📄 Standard reports generated: {self.stats['reports_generated']}")
            
        self.logger.info(f"🕐 Timeline reports generated: {self.stats['timeline_reports_generated']}")
        self.logger.info(f"📋 Master index generated: {'✅ Yes' if index_success else '❌ Failed'}")
        
        if self.stats['errors']:
            self.logger.warning(f"\n⚠️  {len(self.stats['errors'])} errors encountered:")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                self.logger.warning(f"   • {error}")
            if len(self.stats['errors']) > 10:
                self.logger.warning(f"   ... and {len(self.stats['errors']) - 10} more errors")
                
        self.logger.info(f"\n🌐 Open the master index: {self.results_dir / 'index.html'}")
        self.logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Mass regeneration of screenshots and HTML reports for all study material analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--similarity_threshold", 
        type=float,
        help="Override similarity threshold for screenshot extraction (0.0-1.0). Higher = more sensitive to changes."
    )
    
    parser.add_argument(
        "--min_time_between_shots", 
        type=float,
        help="Override minimum time between screenshots in seconds."
    )
    
    parser.add_argument(
        "--skip_screenshots",
        action="store_true",
        help="Skip screenshot regeneration, only regenerate HTML reports."
    )
    
    parser.add_argument(
        "--timeline_only",
        action="store_true", 
        help="Only generate timeline reports (skip standard reports)."
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output."
    )
    
    args = parser.parse_args()
    
    # Create and run mass regenerator
    regenerator = MassRegenerator(
        similarity_threshold=args.similarity_threshold,
        min_time_between_shots=args.min_time_between_shots,
        skip_screenshots=args.skip_screenshots,
        timeline_only=args.timeline_only,
        verbose=args.verbose
    )
    
    regenerator.process_all()

if __name__ == "__main__":
    main()
