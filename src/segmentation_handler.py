\
from typing import List, Tuple, Dict, Any
from pydub import AudioSegment
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SegmentationHandler:
    """
    Handles various audio segmentation strategies.
    """
    def __init__(self, config: Dict):
        self.config = config

    def fixed_time_segmentation(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """
        Create fixed-time segments as a primary segmentation option.
        
        Args:
            audio: AudioSegment object
            
        Returns:
            List of (start, end) tuples for fixed-time segments
        """
        segment_duration = self.config.get('fixed_time_duration', 30000)  # 30 seconds default
        overlap = self.config.get('fixed_time_overlap', 3000)            # 3 seconds overlap default
        
        logger.info(f"Using fixed-time segmentation: {segment_duration/1000}s segments with {overlap/1000}s overlap")
        
        segments = []
        audio_length = len(audio)
        effective_step = segment_duration - overlap  # Step size accounting for overlap
        
        start = 0
        while start < audio_length:
            end = min(start + segment_duration, audio_length)
            
            # Only create segment if it's at least 5 seconds long
            if end - start >= 5000:
                segments.append((start, end))
                logger.debug(f"Fixed-time segment: {start/1000:.1f}s - {end/1000:.1f}s ({(end-start)/1000:.1f}s)")
            
            start += effective_step
            
            # Prevent infinite loop
            if start >= audio_length - 1000:  # Less than 1 second remaining
                break
        
        # Ensure we capture the end of the audio if there's a significant remainder
        if segments and audio_length - segments[-1][1] > 5000: # If remainder is > 5s
            # Ensure the last segment starts 'overlap' ms before the previous one ended to maintain overlap logic
            # Or simply start from the end of the previous segment if non-overlapping logic is preferred for the tail.
            # Current logic: (segments[-1][1] - overlap, audio_length) might re-process part of the last segment.
            # A cleaner way for the final chunk might be:
            # final_chunk_start = segments[-1][1] - overlap if segments[-1][1] - overlap < audio_length else segments[-1][0] # avoid negative or too small start
            # segments.append((final_chunk_start, audio_length))
            # For now, keeping original logic:
            segments.append((segments[-1][1] - overlap, audio_length))

            logger.debug(f"Added final segment: {(segments[-1][1] - overlap)/1000:.1f}s - {audio_length/1000:.1f}s")
        
        total_coverage = sum(end - start for start, end in segments)
        # coverage_percent = (total_coverage / audio_length) * 100 if audio_length > 0 else 0 # Avoid division by zero for empty audio
        
        # logger.info(f"Fixed-time segmentation created {len(segments)} segments")
        # logger.info(f"Coverage: {coverage_percent:.1f}% ({total_coverage/1000:.1f}s / {audio_length/1000:.1f}s)")
        
        return segments

    def create_non_overlapping_segments(
        self, 
        audio_segment_obj: AudioSegment, 
        audio_file_path_str: str, 
        nonsilent_ranges: List[Tuple[int, int]], 
        analysis: Dict # Contains recommended_padding
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Create audio segments WITHOUT overlap to prevent duplicates.
        Uses padding for context but ensures no overlapping between segments.
        
        Args:
            audio_segment_obj: The AudioSegment object of the full audio.
            audio_file_path_str: Path to original audio file (string, for naming segments).
            nonsilent_ranges: List of detected speech ranges.
            analysis: Speech pattern analysis.
            
        Returns:
            List of (segment_file_path, original_start_ms, original_end_ms, padded_start_ms, padded_end_ms)
        """
        total_duration = len(audio_segment_obj)
        
        padding = analysis["recommended_padding"]
        max_length = self.config['max_segment_length']
        min_length = self.config['min_segment_length']
        
        segments = []
        # base_name = Path(audio_file_path_str).stem # Not used with current f-string naming

        MAX_SEGMENTS_ALLOWED = 500
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            if len(segments) >= MAX_SEGMENTS_ALLOWED:
                logger.error(f"EMERGENCY BREAK: Maximum segment limit ({MAX_SEGMENTS_ALLOWED}) reached! Stopping segment generation.")
                break
            segment_length = end - start
            
            if segment_length < min_length:
                if segment_length >= 1000:
                    logger.debug(f"Keeping short but viable segment {i}: {segment_length}ms")
                else:
                    logger.debug(f"Skipping very short segment {i}: {segment_length}ms")
                    continue
            
            if segment_length > max_length:
                chunk_start = start
                chunk_index = 0
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                while chunk_start < end:
                    chunk_end = min(chunk_start + max_length, end)
                    
                    if chunk_end <= chunk_start:
                        logger.warning(f"Segment generation issue: chunk_end ({chunk_end}) <= chunk_start ({chunk_start}). Breaking loop.")
                        break
                    
                    padded_start = max(0, chunk_start - padding)
                    padded_end = min(total_duration, chunk_end + padding)
                    
                    if segments:
                        prev_padded_end = segments[-1][4]
                        if padded_start < prev_padded_end:
                            padded_start = prev_padded_end
                    
                    try:
                        segment_audio = audio_segment_obj[padded_start:padded_end]
                        # Use audio_file_path_str for base name in segment_file
                        segment_file = f"{Path(audio_file_path_str).parent / Path(audio_file_path_str).stem}_{segment_prefix}_{len(segments):03d}_{chunk_index}.wav"
                        segment_audio.export(segment_file, format="wav")
                        
                        segments.append((segment_file, chunk_start, chunk_end, padded_start, padded_end))
                        logger.debug(f"Created segment {chunk_index}: {chunk_start}ms-{chunk_end}ms")
                        
                    except Exception as e:
                        logger.error(f"Failed to create segment {chunk_index}: {e}")
                        break 
                    
                    old_chunk_start = chunk_start
                    chunk_start = chunk_end
                    
                    if chunk_start <= old_chunk_start:
                        logger.error(f"CRITICAL BUG: chunk_start did not advance! old={old_chunk_start}, new={chunk_start}. BREAKING LOOP!")
                        break
                    
                    chunk_index += 1
                    if chunk_index > 1000: 
                        logger.error(f"EMERGENCY BREAK: Too many segments generated ({chunk_index}). Possible infinite loop detected!")
                        break
                    if chunk_start >= end:
                        logger.debug(f"Loop termination: chunk_start ({chunk_start}) >= end ({end})")
                        break
            else:
                padded_start = max(0, start - padding)
                padded_end = min(total_duration, end + padding)
                
                if segments:
                    prev_padded_end = segments[-1][4]
                    if padded_start < prev_padded_end:
                        padded_start = prev_padded_end
                
                segment_prefix = self.config.get('segment_prefix', 'segment')
                segment_audio = audio_segment_obj[padded_start:padded_end]
                segment_file = f"{Path(audio_file_path_str).parent / Path(audio_file_path_str).stem}_{segment_prefix}_{len(segments):03d}.wav"
                segment_audio.export(segment_file, format="wav")
                
                segments.append((segment_file, start, end, padded_start, padded_end))
        
        logger.info(f"Created {len(segments)} NON-OVERLAPPING segments with padding")
        if segments:
            for i, (file, orig_start, orig_end, pad_start, pad_end) in enumerate(segments[:3]):
                logger.info(f"  Segment {i}: {orig_start/1000:.1f}s-{orig_end/1000:.1f}s (padded: {pad_start/1000:.1f}s-{pad_end/1000:.1f}s)")
        return segments

    def create_overlapping_segments(
        self, 
        audio_segment_obj: AudioSegment, 
        audio_file_path_str: str, 
        nonsilent_ranges: List[Tuple[int, int]], 
        analysis: Dict # Contains recommended_padding
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Create audio segments with overlap to prevent cutting off speech.
        
        Args:
            audio_segment_obj: The AudioSegment object of the full audio.
            audio_file_path_str: Path to original audio file (string, for naming segments).
            nonsilent_ranges: List of detected speech ranges.
            analysis: Speech pattern analysis.
            
        Returns:
            List of (segment_file_path, original_start_ms, original_end_ms, padded_start_ms, padded_end_ms)
        """
        total_duration = len(audio_segment_obj)
        
        padding = analysis["recommended_padding"]
        overlap_val = self.config['overlap_duration'] # Renamed from 'overlap' to avoid conflict with arg
        max_length = self.config['max_segment_length']
        min_length = self.config['min_segment_length']
        
        segments = []
        # base_name = Path(audio_file_path_str).stem # Not used with current f-string naming

        for i, (start, end) in enumerate(nonsilent_ranges):
            segment_length = end - start
            
            if segment_length < min_length:
                if segment_length >= 1000:
                    logger.debug(f"Keeping short but viable segment {i}: {segment_length}ms")
                else:
                    if i < len(nonsilent_ranges) - 1:
                        next_start, next_end = nonsilent_ranges[i + 1]
                        gap = next_start - end
                        if gap < self.config['min_silence_len'] * 2:
                            continue
                    logger.debug(f"Skipping very short segment {i}: {segment_length}ms")
                    continue
            
            if segment_length > max_length:
                chunk_start = start
                chunk_index = 0
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                while chunk_start < end:
                    chunk_end = min(chunk_start + max_length, end)
                    
                    if chunk_end <= chunk_start:
                        logger.warning(f"Overlapping segment generation issue: chunk_end ({chunk_end}) <= chunk_start ({chunk_start}). Breaking loop.")
                        break
                    
                    padded_start = max(0, chunk_start - padding)
                    padded_end = min(total_duration, chunk_end + padding)
                    
                    try:
                        segment_audio = audio_segment_obj[padded_start:padded_end]
                        segment_file = f"{Path(audio_file_path_str).parent / Path(audio_file_path_str).stem}_{segment_prefix}_{len(segments):03d}_{chunk_index}.wav"
                        segment_audio.export(segment_file, format="wav")
                        
                        segments.append((segment_file, chunk_start, chunk_end, padded_start, padded_end))
                        logger.debug(f"Created overlapping segment {chunk_index}: {chunk_start}ms-{chunk_end}ms")
                        
                    except Exception as e:
                        logger.error(f"Failed to create overlapping segment {chunk_index}: {e}")
                        break
                    
                    old_chunk_start = chunk_start
                    chunk_start = chunk_end - overlap_val 
                    
                    if chunk_start <= old_chunk_start and chunk_end < end : # check if we are stuck and not at the end
                         logger.warning(f"Chunk start may not be advancing sufficiently if overlap is too large or segment too small. old_chunk_start={old_chunk_start}, chunk_start={chunk_start}, chunk_end={chunk_end}")
                         # Add a more robust break condition if chunk_start doesn't advance meaningfully
                         if chunk_start == old_chunk_start : # Stuck
                            logger.error("CRITICAL BUG: Overlapping chunk_start did not advance at all! Breaking loop.")
                            break


                    chunk_index += 1
                    if chunk_index > 1000:
                        logger.error(f"EMERGENCY BREAK: Too many overlapping segments ({chunk_index}). Breaking loop!")
                        break
                    
                    if chunk_start >= end : # Ensure loop terminates
                        break
            else:
                padded_start = max(0, start - padding)
                padded_end = min(total_duration, end + padding)
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                segment_audio = audio_segment_obj[padded_start:padded_end]
                segment_file = f"{Path(audio_file_path_str).parent / Path(audio_file_path_str).stem}_{segment_prefix}_{len(segments):03d}.wav"
                segment_audio.export(segment_file, format="wav")
                
                segments.append((segment_file, start, end, padded_start, padded_end))
        
        logger.info(f"Created {len(segments)} segments with overlap and padding")
        if segments:
            for i, (file, orig_start, orig_end, pad_start, pad_end) in enumerate(segments[:3]):
                logger.info(f"  Segment {i}: {orig_start/1000:.1f}s-{orig_end/1000:.1f}s (padded: {pad_start/1000:.1f}s-{pad_end/1000:.1f}s)")
        return segments
