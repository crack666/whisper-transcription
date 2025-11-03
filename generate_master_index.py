#!/usr/bin/env python3
"""
Generate Master Index Report

This script creates a comprehensive index page that links to all existing HTML reports
in the results directory. It can be run independently after processing or as part of
the regeneration workflow.

The generated index includes:
- Summary statistics across all processed files
- Links to both standard and timeline reports
- Processing status and error information
- Search and filtering capabilities

Usage:
    python generate_master_index.py [options]
    
Options:
    --output PATH      Custom output path for index.html (default: results/index.html)
    --include_failed   Include failed/incomplete results in the index
    --verbose          Enable verbose logging
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from html_generator import HTMLReportGenerator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class MasterIndexGenerator:
    """Generates a master index page for all study material analysis results"""
    
    def __init__(self, include_failed: bool = False, verbose: bool = False):
        self.logger = setup_logging(verbose)
        self.include_failed = include_failed
        self.results_dir = Path("results")
        self.html_generator = HTMLReportGenerator()
        
    def collect_all_analysis_data(self) -> List[Dict]:
        """Collect analysis data from all results directories"""
        all_analysis_data = []
        
        if not self.results_dir.exists():
            self.logger.error(f"Results directory not found: {self.results_dir}")
            return all_analysis_data
            
        self.logger.info(f"🔍 Scanning for analysis files in: {self.results_dir}")
        
        for subdir in self.results_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Look for analysis.json files
                analysis_pattern = f"{subdir.name}_analysis.json"
                analysis_file = subdir / analysis_pattern
                
                if analysis_file.exists():
                    try:
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                            
                        # Add metadata for index generation
                        analysis_data['_analysis_file_path'] = str(analysis_file)
                        analysis_data['_directory_name'] = subdir.name
                        analysis_data['_has_error'] = bool(analysis_data.get('error'))
                        
                        # Check for existing reports
                        video_name = subdir.name
                        standard_report = subdir / f"{video_name}_report.html"
                        timeline_report = subdir / f"{video_name}_report_TIMELINE.html"
                        
                        analysis_data['_has_standard_report'] = standard_report.exists()
                        analysis_data['_has_timeline_report'] = timeline_report.exists()
                        analysis_data['_standard_report_path'] = str(standard_report.relative_to(self.results_dir)) if standard_report.exists() else None
                        analysis_data['_timeline_report_path'] = str(timeline_report.relative_to(self.results_dir)) if timeline_report.exists() else None
                        
                        # Include based on filter settings
                        if self.include_failed or not analysis_data['_has_error']:
                            all_analysis_data.append(analysis_data)
                            self.logger.debug(f"   ✅ Added: {analysis_file.name}")
                        else:
                            self.logger.debug(f"   ⏭️  Skipped (failed): {analysis_file.name}")
                            
                    except Exception as e:
                        self.logger.warning(f"   ❌ Error loading {analysis_file}: {e}")
                        if self.include_failed:
                            # Add error placeholder
                            error_data = {
                                '_analysis_file_path': str(analysis_file),
                                '_directory_name': subdir.name,
                                '_has_error': True,
                                'error': f"Failed to load analysis file: {e}",
                                '_has_standard_report': False,
                                '_has_timeline_report': False,
                                '_standard_report_path': None,
                                '_timeline_report_path': None
                            }
                            all_analysis_data.append(error_data)
                else:
                    self.logger.debug(f"   ⚠️  No analysis file found in: {subdir.name}")
                    
        self.logger.info(f"📊 Found {len(all_analysis_data)} analysis results")
        return all_analysis_data
        
    def generate_enhanced_index(self, all_analysis_data: List[Dict], output_path: Path) -> bool:
        """Generate an enhanced index page with additional features"""
        try:
            self.logger.info(f"📋 Generating enhanced master index: {output_path}")
            
            # Use the existing HTML generator but enhance the data first
            enhanced_data = self.enhance_analysis_data(all_analysis_data)
            
            self.html_generator.generate_index_page(enhanced_data, str(output_path))
            
            self.logger.info(f"✅ Master index generated successfully")
            self.logger.info(f"🌐 Open in browser: {output_path.absolute()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate master index: {e}")
            return False
            
    def enhance_analysis_data(self, all_analysis_data: List[Dict]) -> List[Dict]:
        """Add metadata and enhancements to analysis data for better index display"""
        enhanced_data = []
        
        for data in all_analysis_data:
            enhanced = data.copy()
            
            # Add report availability information
            if data.get('_has_standard_report') or data.get('_has_timeline_report'):
                enhanced['_report_status'] = 'available'
            elif data.get('_has_error'):
                enhanced['_report_status'] = 'failed'
            else:
                enhanced['_report_status'] = 'missing'
                
            # Calculate processing age
            timestamp_str = data.get('processing_timestamp')
            if timestamp_str:
                try:
                    processing_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    age_days = (datetime.now() - processing_time).days
                    enhanced['_processing_age_days'] = age_days
                except:
                    enhanced['_processing_age_days'] = None
            else:
                enhanced['_processing_age_days'] = None
                
            enhanced_data.append(enhanced)
            
        return enhanced_data
        
    def print_statistics(self, all_analysis_data: List[Dict]) -> None:
        """Print comprehensive statistics about the collected data"""
        total_files = len(all_analysis_data)
        successful_files = len([d for d in all_analysis_data if not d.get('_has_error')])
        failed_files = total_files - successful_files
        
        with_standard_reports = len([d for d in all_analysis_data if d.get('_has_standard_report')])
        with_timeline_reports = len([d for d in all_analysis_data if d.get('_has_timeline_report')])
        
        # Calculate totals
        total_duration = 0
        total_segments = 0
        total_screenshots = 0
        total_pdfs = 0
        
        for data in all_analysis_data:
            if not data.get('_has_error'):
                # Handle nested transcription structure
                transcription_data = data.get("transcription", {})
                if "transcription" in transcription_data:
                    transcription_data = transcription_data["transcription"]
                
                segments = transcription_data.get("segments", [])
                if segments:
                    total_duration += segments[-1].get("end", 0)
                    total_segments += len(segments)
                
                total_screenshots += len(data.get("screenshots", []))
                total_pdfs += len(data.get("related_pdfs", []))
        
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 ANALYSIS COLLECTION STATISTICS")
        self.logger.info("="*60)
        self.logger.info(f"📁 Total analysis files: {total_files}")
        self.logger.info(f"✅ Successful analyses: {successful_files}")
        self.logger.info(f"❌ Failed analyses: {failed_files}")
        self.logger.info(f"📄 Standard reports available: {with_standard_reports}")
        self.logger.info(f"🕐 Timeline reports available: {with_timeline_reports}")
        self.logger.info("")
        self.logger.info(f"⏱️  Total content duration: {total_duration/3600:.1f} hours")
        self.logger.info(f"💬 Total transcript segments: {total_segments}")
        self.logger.info(f"🖼️  Total screenshots: {total_screenshots}")
        self.logger.info(f"📄 Total PDFs: {total_pdfs}")
        self.logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Generate master index page for all study material analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--output", 
        type=Path,
        default=Path("results/index.html"),
        help="Custom output path for index.html (default: results/index.html)"
    )
    
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Include failed/incomplete results in the index"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Create generator and process
    generator = MasterIndexGenerator(
        include_failed=args.include_failed,
        verbose=args.verbose
    )
    
    # Collect all analysis data
    all_analysis_data = generator.collect_all_analysis_data()
    
    if not all_analysis_data:
        generator.logger.error("No analysis data found. Cannot generate index.")
        sys.exit(1)
    
    # Print statistics
    generator.print_statistics(all_analysis_data)
    
    # Generate index
    success = generator.generate_enhanced_index(all_analysis_data, args.output)
    
    if success:
        generator.logger.info(f"\n🎉 Master index generation completed successfully!")
        sys.exit(0)
    else:
        generator.logger.error(f"\n❌ Master index generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
