#!/usr/bin/env python3
"""
Advanced Waveform-Based Speech Detection

Scientific approach using numpy and scipy for precise speech segment detection
based on actual waveform analysis, similar to what you see in Audacity/Premiere.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter
import logging
from typing import List, Tuple, Dict
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaveformSpeechDetector:
    """
    Advanced speech detection using waveform analysis with numpy/scipy.
    Similar to visual waveform analysis in professional audio tools.
    """
    
    def __init__(self, 
                 min_speech_duration_ms: int = 1000,  # Minimum 1 second of speech
                 min_silence_duration_ms: int = 500,   # Minimum 0.5 seconds of silence
                 energy_threshold_factor: float = 0.1,  # 10% of max energy as threshold
                 spectral_rolloff_threshold: float = 0.85):
        
        self.min_speech_duration = min_speech_duration_ms / 1000.0
        self.min_silence_duration = min_silence_duration_ms / 1000.0
        self.energy_threshold_factor = energy_threshold_factor
        self.spectral_rolloff_threshold = spectral_rolloff_threshold
        
    def analyze_audio_file(self, audio_path: str, target_sr: int = 22050) -> Dict:
        """
        Comprehensive waveform analysis of audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate for analysis
            
        Returns:
            Analysis results with speech segments and visualizations
        """
        logger.info(f"🔊 Loading audio file: {audio_path}")
        
        # Load audio with librosa (much better than pydub for analysis)
        y, sr = librosa.load(audio_path, sr=target_sr)
        duration = len(y) / sr
        
        logger.info(f"📊 Audio loaded: {duration:.1f}s, {sr}Hz, {len(y)} samples")
        
        # Perform comprehensive analysis
        analysis = {
            'duration_seconds': duration,
            'sample_rate': sr,
            'samples': len(y),
            'audio_path': audio_path
        }
        
        # 1. Energy-based analysis (like amplitude in waveform)
        energy_segments = self._detect_speech_by_energy(y, sr)
        
        # 2. Spectral analysis (frequency content)
        spectral_segments = self._detect_speech_by_spectral_features(y, sr)
        
        # 3. Combined analysis with refinement
        refined_segments = self._combine_and_refine_segments(
            energy_segments, spectral_segments, duration
        )
        
        # 4. Generate visualization data
        visualization_data = self._generate_visualization_data(y, sr, refined_segments)
        
        analysis.update({
            'energy_segments': energy_segments,
            'spectral_segments': spectral_segments,
            'final_segments': refined_segments,
            'visualization': visualization_data,
            'statistics': self._calculate_statistics(refined_segments, duration)
        })
        
        return analysis
    
    def _detect_speech_by_energy(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments based on energy analysis (similar to waveform amplitude).
        """
        logger.info("🔋 Analyzing energy-based speech detection...")
        
        # Calculate frame-based energy (RMS)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        # RMS energy per frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB for better threshold detection
        rms_db = librosa.amplitude_to_db(rms)
        
        # Adaptive threshold based on signal characteristics
        max_db = np.max(rms_db)
        min_db = np.min(rms_db)
        dynamic_range = max_db - min_db
        
        # Threshold at certain percentage of dynamic range
        threshold_db = min_db + (dynamic_range * self.energy_threshold_factor)
        
        logger.info(f"📊 Energy analysis: max={max_db:.1f}dB, min={min_db:.1f}dB, threshold={threshold_db:.1f}dB")
        
        # Detect speech frames
        speech_frames = rms_db > threshold_db
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sr, hop_length=hop_length)
        
        # Group consecutive speech frames into segments
        segments = self._group_consecutive_frames(speech_frames, frame_times)
        
        logger.info(f"🎯 Energy-based detection found {len(segments)} segments")
        return segments
    
    def _detect_speech_by_spectral_features(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect speech using spectral features (frequency analysis).
        """
        logger.info("🌊 Analyzing spectral-based speech detection...")
        
        # Spectral features
        hop_length = int(0.01 * sr)  # 10ms hop
        
        # Spectral rolloff (frequency below which 85% of energy is concentrated)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Spectral centroid (brightness indicator)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Zero crossing rate (useful for speech vs silence)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        # Combine features for speech detection
        # Speech typically has higher spectral rolloff and centroid than silence
        rolloff_norm = rolloff / np.max(rolloff)
        centroid_norm = centroid / np.max(centroid)
        zcr_norm = zcr / np.max(zcr)
        
        # Combined spectral score
        spectral_score = (rolloff_norm + centroid_norm + zcr_norm) / 3
        
        # Threshold for speech detection
        spectral_threshold = np.percentile(spectral_score, 30)  # Bottom 30% is likely silence
        
        speech_frames = spectral_score > spectral_threshold
        
        # Convert to time segments
        frame_times = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sr, hop_length=hop_length)
        segments = self._group_consecutive_frames(speech_frames, frame_times)
        
        logger.info(f"🎵 Spectral-based detection found {len(segments)} segments")
        return segments
    
    def _group_consecutive_frames(self, speech_frames: np.ndarray, frame_times: np.ndarray) -> List[Tuple[float, float]]:
        """
        Group consecutive TRUE frames into speech segments.
        """
        segments = []
        in_speech = False
        start_time = None
        
        for i, (is_speech, time) in enumerate(zip(speech_frames, frame_times)):
            if is_speech and not in_speech:
                # Start of speech segment
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                if start_time is not None:
                    duration = time - start_time
                    if duration >= self.min_speech_duration:
                        segments.append((start_time, time))
                in_speech = False
                start_time = None
        
        # Handle case where audio ends during speech
        if in_speech and start_time is not None:
            duration = frame_times[-1] - start_time
            if duration >= self.min_speech_duration:
                segments.append((start_time, frame_times[-1]))
        
        return segments
    
    def _combine_and_refine_segments(self, energy_segments: List[Tuple[float, float]], 
                                   spectral_segments: List[Tuple[float, float]], 
                                   total_duration: float) -> List[Tuple[float, float]]:
        """
        Combine energy and spectral analysis results and refine segments.
        """
        logger.info("🔄 Combining and refining segment detection...")
        
        # Combine all segments
        all_segments = energy_segments + spectral_segments
        
        if not all_segments:
            return []
        
        # Sort by start time
        all_segments.sort(key=lambda x: x[0])
        
        # Merge overlapping segments
        merged_segments = []
        current_start, current_end = all_segments[0]
        
        for start, end in all_segments[1:]:
            if start <= current_end + self.min_silence_duration:
                # Merge overlapping or close segments
                current_end = max(current_end, end)
            else:
                # Add completed segment
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last segment
        merged_segments.append((current_start, current_end))
        
        # Filter by minimum duration
        final_segments = [
            (start, end) for start, end in merged_segments
            if (end - start) >= self.min_speech_duration
        ]
        
        logger.info(f"✅ Final refined segments: {len(final_segments)}")
        
        # Log segment details
        for i, (start, end) in enumerate(final_segments):
            logger.info(f"  Segment {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s duration)")
        
        return final_segments
    
    def _generate_visualization_data(self, y: np.ndarray, sr: int, 
                                   segments: List[Tuple[float, float]]) -> Dict:
        """
        Generate data for waveform visualization.
        """
        # Downsample for visualization
        downsample_factor = max(1, len(y) // 10000)  # Max 10k points for visualization
        y_vis = y[::downsample_factor]
        time_vis = np.arange(len(y_vis)) * downsample_factor / sr
        
        return {
            'time_axis': time_vis.tolist(),
            'amplitude': y_vis.tolist(),
            'speech_segments': segments,
            'sample_rate': sr,
            'downsample_factor': downsample_factor
        }
    
    def _calculate_statistics(self, segments: List[Tuple[float, float]], total_duration: float) -> Dict:
        """
        Calculate statistics about detected speech segments.
        """
        if not segments:
            return {
                'total_speech_time': 0,
                'total_silence_time': total_duration,
                'speech_percentage': 0,
                'num_segments': 0,
                'avg_segment_duration': 0,
                'max_gap_duration': 0
            }
        
        total_speech = sum(end - start for start, end in segments)
        total_silence = total_duration - total_speech
        
        segment_durations = [end - start for start, end in segments]
        avg_duration = np.mean(segment_durations)
        
        # Calculate gaps between segments
        gaps = []
        for i in range(len(segments) - 1):
            gap = segments[i+1][0] - segments[i][1]
            gaps.append(gap)
        
        max_gap = max(gaps) if gaps else 0
        
        return {
            'total_speech_time': total_speech,
            'total_silence_time': total_silence,
            'speech_percentage': (total_speech / total_duration) * 100,
            'num_segments': len(segments),
            'avg_segment_duration': avg_duration,
            'max_gap_duration': max_gap,
            'segment_durations': segment_durations
        }

def analyze_waveform_speech(audio_path: str, output_dir: str = "waveform_analysis") -> Dict:
    """
    Main function to analyze speech segments using advanced waveform analysis.
    """
    detector = WaveformSpeechDetector(
        min_speech_duration_ms=2000,  # 2 seconds minimum
        min_silence_duration_ms=1000,  # 1 second silence between segments
        energy_threshold_factor=0.15   # 15% energy threshold
    )
    
    # Perform analysis
    analysis = detector.analyze_audio_file(audio_path)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed analysis
    analysis_file = output_path / "waveform_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Create visualization plot
    create_waveform_plot(analysis, output_path / "waveform_plot.png")
    
    # Print summary
    stats = analysis['statistics']
    segments = analysis['final_segments']
    
    print("\n🔊 WAVEFORM SPEECH ANALYSIS RESULTS")
    print("=" * 50)
    print(f"📁 Audio file: {audio_path}")
    print(f"⏱️  Total duration: {analysis['duration_seconds']:.1f}s")
    print(f"🎯 Speech segments found: {stats['num_segments']}")
    print(f"🗣️  Total speech time: {stats['total_speech_time']:.1f}s ({stats['speech_percentage']:.1f}%)")
    print(f"🔇 Total silence time: {stats['total_silence_time']:.1f}s")
    print(f"📊 Average segment duration: {stats['avg_segment_duration']:.1f}s")
    print(f"⏳ Longest gap between segments: {stats['max_gap_duration']:.1f}s")
    
    print(f"\n📋 DETECTED SEGMENTS:")
    for i, (start, end) in enumerate(segments):
        duration = end - start
        print(f"  {i+1:2d}. {start:6.1f}s - {end:6.1f}s ({duration:5.1f}s)")
    
    print(f"\n💾 Results saved to: {analysis_file}")
    
    return analysis

def create_waveform_plot(analysis: Dict, output_path: str):
    """
    Create a waveform visualization similar to Audacity/Premiere.
    """
    try:
        import matplotlib.pyplot as plt
        
        vis_data = analysis['visualization']
        time_axis = np.array(vis_data['time_axis'])
        amplitude = np.array(vis_data['amplitude'])
        segments = analysis['final_segments']
        
        plt.figure(figsize=(15, 8))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, amplitude, 'b-', alpha=0.7, linewidth=0.5)
        plt.title('Waveform with Detected Speech Segments', fontsize=14)
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Highlight speech segments
        for start, end in segments:
            plt.axvspan(start, end, alpha=0.3, color='green', label='Speech' if start == segments[0][0] else "")
        
        plt.legend()
        
        # Plot energy over time
        plt.subplot(2, 1, 2)
        # Simplified energy representation
        window_size = len(amplitude) // 1000
        if window_size > 1:
            energy = np.array([np.mean(amplitude[i:i+window_size]**2) for i in range(0, len(amplitude), window_size)])
            energy_time = np.array([time_axis[i] for i in range(0, len(time_axis), window_size)])
            plt.plot(energy_time, energy, 'r-', linewidth=1)
        
        plt.title('Energy Levels', fontsize=14)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)
        
        # Highlight speech segments
        for start, end in segments:
            plt.axvspan(start, end, alpha=0.3, color='green')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Waveform plot saved to: {output_path}")
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_waveform_analysis.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    analyze_waveform_speech(audio_file)
