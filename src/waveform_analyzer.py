import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class WaveformAnalyzer:
    """
    Scientific waveform analyzer for precise speech segmentation.
    
    Uses mathematical analysis of audio waveforms to detect speech segments
    with much higher precision than traditional silence detection methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the waveform analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        
        # Analysis parameters
        self.sample_rate = self.config.get('sample_rate', 44100)
        self.frame_size_ms = self.config.get('frame_size_ms', 100)  # 100ms frames
        self.hop_size_ms = self.config.get('hop_size_ms', 50)      # 50ms overlap
        
        # Speech detection thresholds
        self.energy_threshold_factor = self.config.get('energy_threshold_factor', 2.0)
        self.min_speech_duration_ms = self.config.get('min_speech_duration_ms', 1000)  # 1 second
        self.min_silence_duration_ms = self.config.get('min_silence_duration_ms', 2000)  # 2 seconds
        self.volume_percentile_threshold = self.config.get('volume_percentile_threshold', 25)  # 25th percentile
        
        # Advanced parameters
        self.use_spectral_analysis = self.config.get('use_spectral_analysis', False)
        self.adaptive_threshold = self.config.get('adaptive_threshold', True)
        self.merge_close_segments = self.config.get('merge_close_segments', True)
        
        logger.info(f"🔬 Initialized WaveformAnalyzer with scientific parameters")
        logger.info(f"   Frame size: {self.frame_size_ms}ms, Hop size: {self.hop_size_ms}ms")
        logger.info(f"   Min speech: {self.min_speech_duration_ms}ms, Min silence: {self.min_silence_duration_ms}ms")

    def _get_default_config(self) -> Dict:
        """Get default configuration for waveform analysis."""
        return {
            'sample_rate': 44100,
            'frame_size_ms': 100,
            'hop_size_ms': 50,
            'energy_threshold_factor': 2.0,
            'min_speech_duration_ms': 1000,
            'min_silence_duration_ms': 2000,
            'volume_percentile_threshold': 25,
            'use_spectral_analysis': False,
            'adaptive_threshold': True,
            'merge_close_segments': True
        }
    
    def analyze_waveform(self, audio_file: str) -> Dict[str, Any]:
        """
        Perform comprehensive waveform analysis on audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Complete waveform analysis results
        """
        logger.info(f"🔬 Starting scientific waveform analysis: {audio_file}")
        start_time = time.time()
        
        # Load audio
        audio = AudioSegment.from_file(audio_file)
        audio_duration = len(audio) / 1000.0  # Duration in seconds
        
        logger.info(f"📊 Audio loaded: {audio_duration:.1f}s, {audio.frame_rate}Hz, {audio.channels} channel(s)")
        
        # Convert to numpy array for analysis
        audio_data = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            audio_data = audio_data.reshape((-1, 2))
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
        
        # Normalize audio data
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Perform frame-based analysis
        frame_analysis = self._analyze_frames(audio_data, audio.frame_rate)
        
        # Detect speech segments
        speech_segments = self._detect_speech_segments(frame_analysis, audio.frame_rate)
        
        # Post-process segments
        processed_segments = self._post_process_segments(speech_segments, audio_duration)
        
        # Calculate statistics
        stats = self._calculate_statistics(processed_segments, audio_duration, frame_analysis)
        
        analysis_time = time.time() - start_time
        
        result = {
            'audio_file': audio_file,
            'audio_duration_seconds': audio_duration,
            'sample_rate': audio.frame_rate,
            'speech_segments': processed_segments,  # List of (start_ms, end_ms) tuples
            'frame_analysis': frame_analysis,
            'statistics': stats,
            'analysis_time_seconds': analysis_time,
            'config_used': self.config.copy()
        }
        
        logger.info(f"✅ Waveform analysis completed in {analysis_time:.2f}s")
        logger.info(f"🎯 Detected {len(processed_segments)} speech segments")
        logger.info(f"📊 Speech coverage: {stats['speech_coverage_percent']:.1f}%")
        
        return result
    
    def _analyze_frames(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        Analyze audio in frames to extract features.
        
        Args:
            audio_data: Normalized audio data
            sample_rate: Audio sample rate
            
        Returns:
            Frame-based analysis results
        """
        frame_size_samples = int(self.frame_size_ms * sample_rate / 1000)
        hop_size_samples = int(self.hop_size_ms * sample_rate / 1000)
        
        num_frames = (len(audio_data) - frame_size_samples) // hop_size_samples + 1
        
        # Initialize feature arrays
        energy = np.zeros(num_frames)
        rms = np.zeros(num_frames)
        zcr = np.zeros(num_frames)  # Zero crossing rate
        
        logger.debug(f"🔍 Analyzing {num_frames} frames (frame_size: {frame_size_samples}, hop: {hop_size_samples})")
        
        for i in range(num_frames):
            start_idx = i * hop_size_samples
            end_idx = start_idx + frame_size_samples
            frame = audio_data[start_idx:end_idx]
            
            # Energy (sum of squared amplitudes)
            energy[i] = np.sum(frame ** 2)
            
            # RMS (Root Mean Square)
            rms[i] = np.sqrt(np.mean(frame ** 2))
            
            # Zero Crossing Rate (measure of spectral content)
            if len(frame) > 1:
                zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))
        
        # Calculate time axis for frames
        time_axis = np.arange(num_frames) * self.hop_size_ms / 1000.0
        
        return {
            'time_axis': time_axis,
            'energy': energy,
            'rms': rms,
            'zcr': zcr,
            'frame_size_ms': self.frame_size_ms,
            'hop_size_ms': self.hop_size_ms
        }
    
    def _detect_speech_segments(self, frame_analysis: Dict, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments using scientific thresholds.
        
        Args:
            frame_analysis: Results from frame analysis
            sample_rate: Audio sample rate
            
        Returns:
            List of (start_seconds, end_seconds) tuples
        """
        energy = frame_analysis['energy']
        rms = frame_analysis['rms']
        time_axis = frame_analysis['time_axis']
        
        # Calculate adaptive thresholds
        if self.adaptive_threshold:
            # Use percentile-based threshold for robustness
            energy_threshold = np.percentile(energy[energy > 0], self.volume_percentile_threshold)
            rms_threshold = np.percentile(rms[rms > 0], self.volume_percentile_threshold)
        else:
            # Use mean-based threshold
            energy_threshold = np.mean(energy) * self.energy_threshold_factor
            rms_threshold = np.mean(rms) * self.energy_threshold_factor
        
        logger.debug(f"🎯 Thresholds - Energy: {energy_threshold:.6f}, RMS: {rms_threshold:.6f}")
        
        # Detect speech frames (both energy and RMS above threshold)
        speech_frames = (energy > energy_threshold) & (rms > rms_threshold)
        
        # Find speech segments
        segments = []
        in_speech = False
        segment_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Start of speech segment
                segment_start = time_axis[i]
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                segment_end = time_axis[i]
                segment_duration = segment_end - segment_start
                
                # Only keep segments longer than minimum duration
                if segment_duration >= self.min_speech_duration_ms / 1000.0:
                    segments.append((segment_start, segment_end))
                
                in_speech = False
        
        # Handle case where audio ends during speech
        if in_speech:
            segment_end = time_axis[-1] + self.hop_size_ms / 1000.0
            segment_duration = segment_end - segment_start
            if segment_duration >= self.min_speech_duration_ms / 1000.0:
                segments.append((segment_start, segment_end))
        
        logger.debug(f"🎤 Initial detection: {len(segments)} speech segments")
        return segments
    
    def _post_process_segments(self, segments: List[Tuple[float, float]], 
                             audio_duration: float) -> List[Tuple[int, int]]:
        """
        Post-process detected segments for better quality.
        
        Args:
            segments: List of (start_seconds, end_seconds) tuples
            audio_duration: Total audio duration in seconds
            
        Returns:
            List of (start_ms, end_ms) tuples
        """
        if not segments:
            return []
        
        processed = []
        
        # Merge close segments if enabled
        if self.merge_close_segments:
            merged_segments = []
            current_start, current_end = segments[0]
            
            for start, end in segments[1:]:
                gap_duration = start - current_end
                
                # Merge if gap is shorter than minimum silence duration
                if gap_duration < self.min_silence_duration_ms / 1000.0:
                    current_end = end  # Extend current segment
                    logger.debug(f"🔗 Merged segments with {gap_duration*1000:.0f}ms gap")
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_segments.append((current_start, current_end))
            segments = merged_segments
        
        # Convert to milliseconds and ensure bounds
        for start_sec, end_sec in segments:
            start_ms = max(0, int(start_sec * 1000))
            end_ms = min(int(audio_duration * 1000), int(end_sec * 1000))
            
            # Final duration check
            if end_ms - start_ms >= self.min_speech_duration_ms:
                processed.append((start_ms, end_ms))
        
        logger.info(f"✂️ Post-processing: {len(segments)} -> {len(processed)} segments")
        return processed
    
    def _calculate_statistics(self, segments: List[Tuple[int, int]], 
                            audio_duration: float, frame_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the analysis.
        
        Args:
            segments: Final speech segments in milliseconds
            audio_duration: Total audio duration in seconds
            frame_analysis: Frame analysis results
            
        Returns:
            Statistical analysis results
        """
        if not segments:
            return {
                'speech_coverage_percent': 0.0,
                'total_speech_duration_seconds': 0.0,
                'num_segments': 0,
                'avg_segment_duration_seconds': 0.0,
                'longest_segment_seconds': 0.0,
                'shortest_segment_seconds': 0.0,
                'longest_silence_gap_seconds': 0.0
            }
        
        # Calculate speech coverage
        total_speech_duration = sum(end - start for start, end in segments) / 1000.0
        speech_coverage = (total_speech_duration / audio_duration) * 100
        
        # Segment statistics
        segment_durations = [(end - start) / 1000.0 for start, end in segments]
        avg_segment_duration = np.mean(segment_durations)
        longest_segment = np.max(segment_durations)
        shortest_segment = np.min(segment_durations)
        
        # Silence gap analysis
        silence_gaps = []
        for i in range(len(segments) - 1):
            gap_start = segments[i][1]
            gap_end = segments[i + 1][0]
            gap_duration = (gap_end - gap_start) / 1000.0
            silence_gaps.append(gap_duration)
        
        longest_silence = np.max(silence_gaps) if silence_gaps else 0.0
        
        # Energy statistics
        energy = frame_analysis['energy']
        energy_stats = {
            'mean_energy': float(np.mean(energy)),
            'std_energy': float(np.std(energy)),
            'max_energy': float(np.max(energy)),
            'energy_dynamic_range_db': float(20 * np.log10(np.max(energy) / np.mean(energy[energy > 0]))) if np.any(energy > 0) else 0.0
        }
        
        return {
            'speech_coverage_percent': speech_coverage,
            'total_speech_duration_seconds': total_speech_duration,
            'num_segments': len(segments),
            'avg_segment_duration_seconds': avg_segment_duration,
            'longest_segment_seconds': longest_segment,
            'shortest_segment_seconds': shortest_segment,
            'longest_silence_gap_seconds': longest_silence,
            'energy_statistics': energy_stats
        }
    
    def create_visualization(self, analysis_result: Dict, output_path: Optional[str] = None) -> str:
        """
        Create visualization of waveform analysis results.
        
        Args:
            analysis_result: Results from analyze_waveform()
            output_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization
        """
        if output_path is None:
            audio_file = analysis_result['audio_file']
            output_path = f"{Path(audio_file).stem}_waveform_analysis.png"
        
        frame_analysis = analysis_result['frame_analysis']
        segments = analysis_result['speech_segments']
        
        # Create subplot layout
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        time_axis = frame_analysis['time_axis']
        
        # Plot 1: Energy
        axes[0].plot(time_axis, frame_analysis['energy'], 'b-', alpha=0.7, label='Energy')
        axes[0].set_ylabel('Energy')
        axes[0].set_title('Waveform Analysis - Energy')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: RMS
        axes[1].plot(time_axis, frame_analysis['rms'], 'g-', alpha=0.7, label='RMS')
        axes[1].set_ylabel('RMS')
        axes[1].set_title('RMS (Root Mean Square)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Zero Crossing Rate
        axes[2].plot(time_axis, frame_analysis['zcr'], 'r-', alpha=0.7, label='Zero Crossing Rate')
        axes[2].set_ylabel('ZCR')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_title('Zero Crossing Rate')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Highlight detected speech segments on all plots
        for start_ms, end_ms in segments:
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            for ax in axes:
                ax.axvspan(start_sec, end_sec, alpha=0.2, color='yellow', label='Speech' if segments.index((start_ms, end_ms)) == 0 else "")
        
        # Add legend for speech segments (only once)
        if segments:
            axes[0].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualization saved: {output_path}")
        return output_path
    
    def export_segments_for_transcription(self, analysis_result: Dict) -> List[Tuple[int, int]]:
        """
        Export segments in format compatible with enhanced_transcriber.
        
        Args:
            analysis_result: Results from analyze_waveform()
            
        Returns:
            List of (start_ms, end_ms) tuples ready for transcription
        """
        segments = analysis_result['speech_segments']
        
        logger.info(f"📤 Exporting {len(segments)} segments for transcription")
        logger.info(f"🎯 Coverage: {analysis_result['statistics']['speech_coverage_percent']:.1f}%")
        
        return segments

    def detect_speech_segments(self, audio_segment: AudioSegment) -> List[Tuple[int, int]]:
        """
        Detects speech segments from an AudioSegment object using waveform analysis.
        This is the primary method for precision_waveform_detection.

        Args:
            audio_segment: The pydub AudioSegment object to analyze.

        Returns:
            A list of tuples, where each tuple is (start_ms, end_ms) of a detected speech segment.
        """
        logger.info(f"🔬 Starting scientific waveform analysis for precision_waveform_detection")
        start_time = time.time()

        audio_duration_ms = len(audio_segment)
        audio_duration_s = audio_duration_ms / 1000.0

        logger.info(f"📊 Audio loaded: {audio_duration_s:.1f}s, {audio_segment.frame_rate}Hz, {audio_segment.channels} channel(s)")

        # Convert to numpy array for analysis
        audio_data = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape((-1, 2))
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
        
        # Normalize audio data
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0: # Avoid division by zero for silent audio
            audio_data = audio_data / np.max(np.abs(audio_data))
        else: # Handle completely silent audio
            logger.info("Audio data is completely silent. No speech segments will be detected.")
            return []

        # Perform frame-based analysis
        # Note: _analyze_frames expects sample_rate, ensure audio_segment.frame_rate is correct
        frame_analysis = self._analyze_frames(audio_data, audio_segment.frame_rate)
        
        # Detect speech segments (returns start_sec, end_sec)
        # Note: _detect_speech_segments expects sample_rate
        speech_segments_sec = self._detect_speech_segments(frame_analysis, audio_segment.frame_rate)
        
        # Post-process segments (returns start_ms, end_ms)
        # Note: _post_process_segments expects audio_duration in seconds
        processed_segments_ms = self._post_process_segments(speech_segments_sec, audio_duration_s)
        
        analysis_time = time.time() - start_time
        
        logger.info(f"✅ Waveform analysis for speech detection completed in {analysis_time:.2f}s")
        logger.info(f"🎯 Detected {len(processed_segments_ms)} speech segments via detect_speech_segments")
        
        return processed_segments_ms

    def defensive_silence_detection(
        self, 
        audio: AudioSegment, 
        min_silence_len: int, 
        silence_thresh_offset: float, 
        padding_ms: int,
        analysis: Optional[Dict] = None # Added analysis for consistency, can be used for dynamic thresholding
    ) -> List[Tuple[int, int]]:
        """
        Detects non-silent (likely speech) ranges using pydub's silence detection,
        with configurable parameters for a "defensive" approach.

        Args:
            audio: The AudioSegment to analyze.
            min_silence_len: Minimum duration of silence to be considered a split point (in ms).
            silence_thresh_offset: Offset from the audio's mean dBFS to set the silence threshold.
                                   A positive value makes it more tolerant to noise (detects more as speech).
                                   A negative value makes it stricter (detects less as speech).
            padding_ms: Milliseconds of padding to add to the start and end of detected speech segments.
            analysis: Optional speech pattern analysis from EnhancedAudioTranscriber, could be used for dynamic thresholds.

        Returns:
            A list of (start_ms, end_ms) tuples for detected non-silent ranges.
        """
        from pydub.silence import detect_nonsilent # Local import to keep pydub.silence optional if not used elsewhere

        # Determine silence threshold. If analysis is provided and contains mean_volume_db, use it.
        # Otherwise, use the audio segment's own dBFS.
        if analysis and 'mean_volume_db' in analysis:
            reference_dbfs = analysis['mean_volume_db']
            logger.info(f"Using mean_volume_db from analysis ({reference_dbfs:.2f} dBFS) for silence threshold.")
        else:
            reference_dbfs = audio.dBFS
            logger.info(f"Using audio.dBFS ({reference_dbfs:.2f} dBFS) for silence threshold (analysis not provided or no mean_volume_db).")

        # Ensure reference_dbfs is a finite number, fallback if it's -inf (completely silent audio)
        if not np.isfinite(reference_dbfs):
            logger.warning(f"Reference dBFS is not finite ({reference_dbfs}). Falling back to -50 dBFS for threshold calculation.")
            reference_dbfs = -50.0 # A reasonable fallback for silent audio

        silence_threshold = reference_dbfs - silence_thresh_offset 
        logger.info(f"Defensive silence detection: min_silence_len={min_silence_len}ms, silence_thresh={silence_threshold:.2f}dB (offset={silence_thresh_offset}dB), padding={padding_ms}ms")

        # detect_nonsilent returns a list of [start, end] in ms
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_threshold,
            seek_step=1 # Check every 1 ms
        )

        if not nonsilent_ranges:
            logger.info("No non-silent ranges detected by defensive_silence_detection.")
            return []

        # Apply padding and ensure segments are within audio bounds
        padded_segments = []
        audio_len_ms = len(audio)
        for start_ms, end_ms in nonsilent_ranges:
            padded_start = max(0, start_ms - padding_ms)
            padded_end = min(audio_len_ms, end_ms + padding_ms)
            
            # Ensure segment has a minimum length after padding (e.g., > 0 or a configured min)
            if padded_end > padded_start:
                 # Optional: Merge overlapping segments after padding, though detect_nonsilent usually gives distinct chunks.
                 # For simplicity, direct addition here. More complex merging could be added if needed.
                if padded_segments and padded_start < padded_segments[-1][1]: # Overlap with previous
                    # Merge with the previous segment
                    logger.debug(f"Merging overlapping segment after padding: prev_end={padded_segments[-1][1]}, curr_start={padded_start}")
                    padded_segments[-1] = (padded_segments[-1][0], max(padded_segments[-1][1], padded_end))
                else:
                    padded_segments.append((padded_start, padded_end))

        # Filter out very short segments that might have resulted from padding or were inherently small
        min_segment_len_config = self.config.get('min_segment_length', 1000) # Get from config, default 1s
        final_segments = [(s, e) for s, e in padded_segments if (e - s) >= min_segment_len_config]


        logger.info(f"Defensive silence detection found {len(nonsilent_ranges)} raw non-silent ranges, "
                    f"{len(padded_segments)} after padding, {len(final_segments)} final segments.")
        return final_segments

def create_waveform_analyzer(config: Optional[Dict] = None) -> WaveformAnalyzer:
    """
    Factory function to create a WaveformAnalyzer instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured WaveformAnalyzer instance
    """
    return WaveformAnalyzer(config)


# Example configurations for different use cases
PRECISION_CONFIG = {
    'frame_size_ms': 50,  # Smaller frames for higher precision
    'hop_size_ms': 25,    # More overlap
    'min_speech_duration_ms': 500,  # Shorter minimum segments
    'min_silence_duration_ms': 1000,  # Shorter silence requirement
    'volume_percentile_threshold': 20,  # More sensitive threshold
    'adaptive_threshold': True,
    'merge_close_segments': True
}

CONSERVATIVE_CONFIG = {
    'frame_size_ms': 200,  # Larger frames for stability
    'hop_size_ms': 100,    # Less overlap
    'min_speech_duration_ms': 2000,  # Longer minimum segments
    'min_silence_duration_ms': 3000,  # Longer silence requirement
    'volume_percentile_threshold': 30,  # Less sensitive threshold
    'adaptive_threshold': True,
    'merge_close_segments': True
}

LECTURE_CONFIG = {
    'frame_size_ms': 100,  # Balanced for lectures
    'hop_size_ms': 50,     # Standard overlap
    'min_speech_duration_ms': 1000,  # 1 second minimum
    'min_silence_duration_ms': 2000,  # 2 second silence
    'volume_percentile_threshold': 25,  # Balanced threshold
    'adaptive_threshold': True,
    'merge_close_segments': True
}
