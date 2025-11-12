#!/usr/bin/env python3
"""
Transcript Text Extractor

Extracts pure transcript text from analysis.json files for further text processing,
analysis, or external tool integration.

Features:
- Extract from single analysis file or batch process entire results directory
- Multiple output formats: plain text, timestamped, segmented
- Optional metadata inclusion (timestamps, speaker info, confidence scores)
- UTF-8 encoding for international characters
- Flexible output naming

Usage:
    # Single file extraction
    python extract_transcript_text.py --input results/video_name/video_name_analysis.json
    
    # Batch extraction from results directory
    python extract_transcript_text.py --batch --input results/
    
    # With timestamps
    python extract_transcript_text.py --input analysis.json --timestamps
    
    # With segment numbering
    python extract_transcript_text.py --input analysis.json --segments
    
    # Custom output path
    python extract_transcript_text.py --input analysis.json --output transcript.txt
    
    # Include metadata (confidence, duration)
    python extract_transcript_text.py --input analysis.json --metadata
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriptExtractor:
    """Extracts and formats transcript text from analysis JSON files"""
    
    def __init__(self, include_timestamps: bool = False, 
                 include_segments: bool = False,
                 include_metadata: bool = False):
        self.include_timestamps = include_timestamps
        self.include_segments = include_segments
        self.include_metadata = include_metadata
        
    def extract_from_file(self, analysis_file: Path) -> Optional[Tuple[str, Dict]]:
        """
        Extract transcript text from a single analysis file.
        
        Args:
            analysis_file: Path to the analysis JSON file
            
        Returns:
            Tuple of (extracted_text, metadata) or None if failed
        """
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract segments from nested structure
            segments = self._extract_segments(data)
            
            if not segments:
                logger.warning(f"No transcript segments found in {analysis_file.name}")
                return None
                
            # Generate formatted text
            text = self._format_transcript(segments)
            
            # Extract metadata
            metadata = self._extract_metadata(data, segments)
            
            return text, metadata
            
        except FileNotFoundError:
            logger.error(f"File not found: {analysis_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {analysis_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {analysis_file}: {e}")
            return None
            
    def _extract_segments(self, data: Dict) -> List[Dict]:
        """Extract transcript segments from various JSON structures"""
        segments = []
        
        # Try different possible structures
        if 'transcription' in data:
            transcription = data['transcription']
            
            # Nested structure: data['transcription']['transcription']['segments']
            if isinstance(transcription, dict) and 'transcription' in transcription:
                segments = transcription['transcription'].get('segments', [])
            # Direct structure: data['transcription']['segments']
            elif isinstance(transcription, dict) and 'segments' in transcription:
                segments = transcription['segments']
            # List structure: data['transcription'] is already segments
            elif isinstance(transcription, list):
                segments = transcription
                
        return segments
        
    def _format_transcript(self, segments: List[Dict]) -> str:
        """Format transcript segments into readable text"""
        lines = []
        
        for idx, segment in enumerate(segments, 1):
            line_parts = []
            
            # Segment number
            if self.include_segments:
                line_parts.append(f"[{idx}]")
                
            # Timestamp
            if self.include_timestamps:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                timestamp = self._format_timestamp(start)
                line_parts.append(f"[{timestamp}]")
                
            # Text content
            text = segment.get('text', '').strip()
            line_parts.append(text)
            
            # Metadata (confidence, etc.)
            if self.include_metadata:
                confidence = segment.get('avg_logprob', 0)
                # Convert log probability to percentage (approximate)
                confidence_pct = min(100, max(0, int((confidence + 2) * 50)))
                line_parts.append(f"(conf: {confidence_pct}%)")
                
            lines.append(' '.join(line_parts))
            
        return '\n'.join(lines)
        
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS or MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
            
    def _extract_metadata(self, data: Dict, segments: List[Dict]) -> Dict:
        """Extract useful metadata from analysis data"""
        metadata = {
            'total_segments': len(segments),
            'video_path': data.get('video_path', 'Unknown'),
            'processing_timestamp': data.get('processing_timestamp', 'Unknown'),
        }
        
        if segments:
            # Calculate total duration
            last_segment = segments[-1]
            metadata['total_duration_seconds'] = last_segment.get('end', 0)
            metadata['total_duration_formatted'] = self._format_timestamp(
                metadata['total_duration_seconds']
            )
            
            # Word count (approximate)
            total_words = sum(len(s.get('text', '').split()) for s in segments)
            metadata['approximate_word_count'] = total_words
            
            # Average confidence
            confidences = [s.get('avg_logprob', 0) for s in segments if 'avg_logprob' in s]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                metadata['average_confidence'] = f"{min(100, max(0, int((avg_conf + 2) * 50)))}%"
                
        return metadata
        
    def save_to_file(self, text: str, output_path: Path, metadata: Optional[Dict] = None):
        """Save extracted text to file with optional metadata header"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write metadata header if requested
                if self.include_metadata and metadata:
                    f.write("=" * 70 + "\n")
                    f.write("TRANSCRIPT METADATA\n")
                    f.write("=" * 70 + "\n")
                    for key, value in metadata.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    f.write("=" * 70 + "\n\n")
                    
                # Write transcript text
                f.write(text)
                
            logger.info(f"✅ Transcript saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transcript to {output_path}: {e}")
            return False


def process_single_file(args):
    """Process a single analysis file"""
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False
        
    # Create extractor
    extractor = TranscriptExtractor(
        include_timestamps=args.timestamps,
        include_segments=args.segments,
        include_metadata=args.metadata
    )
    
    # Extract transcript
    result = extractor.extract_from_file(input_path)
    if not result:
        return False
        
    text, metadata = result
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output name
        output_name = input_path.stem.replace('_analysis', '') + '_transcript.txt'
        output_path = input_path.parent / output_name
        
    # Save to file
    success = extractor.save_to_file(text, output_path, metadata)
    
    # Print summary
    if success and metadata:
        logger.info(f"📊 Summary:")
        logger.info(f"   Segments: {metadata.get('total_segments', 'N/A')}")
        logger.info(f"   Duration: {metadata.get('total_duration_formatted', 'N/A')}")
        logger.info(f"   Words: ~{metadata.get('approximate_word_count', 'N/A')}")
        
    return success


def process_batch(args):
    """Process all analysis files in a directory"""
    input_dir = Path(args.input)
    
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return False
        
    # Find all analysis files
    analysis_files = list(input_dir.glob("*/*_analysis.json"))
    
    if not analysis_files:
        logger.warning(f"No analysis files found in {input_dir}")
        return False
        
    logger.info(f"📂 Found {len(analysis_files)} analysis files to process")
    
    # Create extractor
    extractor = TranscriptExtractor(
        include_timestamps=args.timestamps,
        include_segments=args.segments,
        include_metadata=args.metadata
    )
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, analysis_file in enumerate(analysis_files, 1):
        logger.info(f"\n[{i}/{len(analysis_files)}] Processing: {analysis_file.parent.name}")
        
        result = extractor.extract_from_file(analysis_file)
        if not result:
            failed += 1
            continue
            
        text, metadata = result
        
        # Auto-generate output path
        output_name = analysis_file.stem.replace('_analysis', '') + '_transcript.txt'
        output_path = analysis_file.parent / output_name
        
        if extractor.save_to_file(text, output_path, metadata):
            successful += 1
        else:
            failed += 1
            
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"🎉 Batch processing completed!")
    logger.info(f"{'='*70}")
    logger.info(f"✅ Successfully extracted: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Total processed: {len(analysis_files)}")
    logger.info(f"{'='*70}")
    
    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract pure transcript text from analysis JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input analysis JSON file or directory (for batch mode)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output text file path (single file mode only). Auto-generated if not specified.'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Batch mode: process all analysis files in input directory'
    )
    
    parser.add_argument(
        '--timestamps', '-t',
        action='store_true',
        help='Include timestamps for each segment'
    )
    
    parser.add_argument(
        '--segments', '-s',
        action='store_true',
        help='Include segment numbers'
    )
    
    parser.add_argument(
        '--metadata', '-m',
        action='store_true',
        help='Include metadata header (duration, word count, confidence)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Process
    if args.batch:
        success = process_batch(args)
    else:
        success = process_single_file(args)
        
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
