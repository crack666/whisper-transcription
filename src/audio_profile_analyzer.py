#!/usr/bin/env python3
"""
Audio Profile Analyzer for Whisper Transcription Optimization.

This module handles audio analysis, similarity calculations, and profile matching
to determine optimal transcription settings based on audio characteristics.
"""

import os
import logging
import hashlib
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

from .utils import extract_audio_from_video

@dataclass
class AudioProfile:
    """Audio characteristics profile for optimization."""
    duration_seconds: float
    mean_volume_db: float
    volume_std: float
    silence_ratio: float
    speech_ratio: float
    speaker_type: str
    estimated_pause_length: float
    background_noise_level: float
    dynamic_range: float
    file_hash: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AudioProfile':
        return cls(**data)

@dataclass
class OptimizationResult:
    """Result of a configuration optimization test."""
    config_name: str
    parameters: Dict
    word_count: int
    char_count: int
    segment_count: int
    processing_time: float
    success: bool
    quality_score: float
    audio_profile: AudioProfile
    timestamp: str
    
    def to_dict(self) -> Dict:
        result_dict = asdict(self)
        result_dict['audio_profile'] = self.audio_profile.to_dict()
        return result_dict

class AudioProfileAnalyzer:
    """Analyzes audio files and calculates profile similarities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Check for required audio libraries
        if librosa is None:
            self.logger.warning("librosa not available - using basic audio analysis")
        if sf is None:
            self.logger.warning("soundfile not available - using basic audio analysis")
    
    def analyze_audio_profile(self, audio_file: str) -> AudioProfile:
        """
        Analyze audio file characteristics to create optimization profile.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            AudioProfile with analyzed characteristics
        """
        self.logger.info(f"Analyzing audio profile for: {audio_file}")
        
        # Convert video to audio if needed
        if audio_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.logger.info("Converting video to audio for analysis...")
            audio_file = extract_audio_from_video(audio_file)
        
        # Calculate file hash for caching/identification
        file_hash = self._calculate_file_hash(audio_file)
        
        try:
            if librosa is not None and sf is not None:
                return self._analyze_with_librosa(audio_file, file_hash)
            else:
                return self._analyze_basic(audio_file, file_hash)
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return self._create_fallback_profile(audio_file, file_hash)
    
    def _analyze_with_librosa(self, audio_file: str, file_hash: str) -> AudioProfile:
        """Detailed audio analysis using librosa."""
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_file, sr=None)
            duration = len(y) / sr
            
            # Volume analysis
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            mean_volume_db = 20 * np.log10(mean_rms) if mean_rms > 0 else -60
            volume_std = np.std(20 * np.log10(rms + 1e-8))
            
            # Dynamic range
            max_db = 20 * np.log10(np.max(np.abs(y)) + 1e-8)
            min_db = 20 * np.log10(np.percentile(np.abs(y), 5) + 1e-8)
            dynamic_range = max_db - min_db
            
            # Silence analysis
            silence_threshold = np.percentile(rms, 20)
            silence_frames = rms < silence_threshold
            silence_ratio = np.sum(silence_frames) / len(silence_frames)
            
            # Speech activity detection
            speech_ratio = 1.0 - silence_ratio
            
            # Estimate pause lengths
            silence_segments = self._find_silence_segments(silence_frames, sr)
            avg_pause_length = np.mean(silence_segments) if silence_segments else 0.5
            
            # Background noise estimation
            noise_floor = np.percentile(rms, 10)
            background_noise_level = 20 * np.log10(noise_floor + 1e-8)
            
            # Determine speaker type based on analysis
            speaker_type = self._classify_speaker_type(speech_ratio, avg_pause_length, dynamic_range)
            
            return AudioProfile(
                duration_seconds=duration,
                mean_volume_db=float(mean_volume_db),
                volume_std=float(volume_std),
                silence_ratio=float(silence_ratio),
                speech_ratio=float(speech_ratio),
                speaker_type=speaker_type,
                estimated_pause_length=float(avg_pause_length),
                background_noise_level=float(background_noise_level),
                dynamic_range=float(dynamic_range),
                file_hash=file_hash
            )
            
        except Exception as e:
            self.logger.error(f"Librosa analysis failed: {e}")
            return self._analyze_basic(audio_file, file_hash)
    
    def _analyze_basic(self, audio_file: str, file_hash: str) -> AudioProfile:
        """Basic audio analysis without librosa."""
        try:
            # Get basic file info
            file_size = os.path.getsize(audio_file)
            
            # Estimate duration (rough approximation)
            # Assume average bitrate for estimation
            estimated_duration = max(10.0, file_size / (128 * 1024 / 8))  # Assume 128kbps
            
            # Create basic profile with reasonable defaults
            return AudioProfile(
                duration_seconds=estimated_duration,
                mean_volume_db=-20.0,  # Reasonable default
                volume_std=8.0,
                silence_ratio=0.3,  # 30% silence is common
                speech_ratio=0.7,   # 70% speech
                speaker_type="normal",
                estimated_pause_length=1.0,
                background_noise_level=-40.0,
                dynamic_range=30.0,
                file_hash=file_hash
            )
            
        except Exception as e:
            self.logger.error(f"Basic analysis failed: {e}")
            return self._create_fallback_profile(audio_file, file_hash)
    
    def _create_fallback_profile(self, audio_file: str, file_hash: str) -> AudioProfile:
        """Create fallback profile when analysis fails."""
        return AudioProfile(
            duration_seconds=60.0,
            mean_volume_db=-25.0,
            volume_std=10.0,
            silence_ratio=0.4,
            speech_ratio=0.6,
            speaker_type="unknown",
            estimated_pause_length=1.5,
            background_noise_level=-45.0,
            dynamic_range=25.0,
            file_hash=file_hash
        )
    
    def _find_silence_segments(self, silence_frames: np.ndarray, sr: int) -> List[float]:
        """Find silence segment durations."""
        segments = []
        in_silence = False
        start_frame = 0
        
        hop_length = 512  # Default librosa hop length
        frame_duration = hop_length / sr
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                start_frame = i
                in_silence = True
            elif not is_silent and in_silence:
                duration = (i - start_frame) * frame_duration
                segments.append(duration)
                in_silence = False
        
        return segments
    
    def _classify_speaker_type(self, speech_ratio: float, avg_pause_length: float, dynamic_range: float) -> str:
        """Classify speaker type based on characteristics."""
        if speech_ratio < 0.4:
            return "very_sparse"
        elif speech_ratio < 0.6:
            return "sparse"
        elif avg_pause_length > 2.0:
            return "slow"
        elif dynamic_range > 35:
            return "expressive"
        else:
            return "normal"
    
    def _calculate_file_hash(self, audio_file: str) -> str:
        """Calculate MD5 hash of audio file for identification."""
        try:
            with open(audio_file, 'rb') as f:
                # Read first and last 64KB for efficiency
                start_chunk = f.read(65536)
                f.seek(-65536, 2)
                end_chunk = f.read(65536)
                
            return hashlib.md5(start_chunk + end_chunk).hexdigest()
        except Exception:
            return hashlib.md5(audio_file.encode()).hexdigest()
    
    def calculate_similarity_score(self, profile1: AudioProfile, profile2: AudioProfile) -> float:
        """
        Calculate similarity score between two audio profiles.
        
        Args:
            profile1: First audio profile
            profile2: Second audio profile
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Weighted similarity calculation
        weights = {
            'speech_ratio': 0.3,
            'speaker_type': 0.25,
            'silence_ratio': 0.2,
            'estimated_pause_length': 0.15,
            'dynamic_range': 0.1
        }
        
        total_score = 0.0
        
        # Speech ratio similarity
        speech_diff = abs(profile1.speech_ratio - profile2.speech_ratio)
        speech_score = max(0, 1 - speech_diff * 2)  # Scale to 0-1
        total_score += speech_score * weights['speech_ratio']
        
        # Speaker type similarity
        speaker_score = 1.0 if profile1.speaker_type == profile2.speaker_type else 0.3
        total_score += speaker_score * weights['speaker_type']
        
        # Silence ratio similarity
        silence_diff = abs(profile1.silence_ratio - profile2.silence_ratio)
        silence_score = max(0, 1 - silence_diff * 2)
        total_score += silence_score * weights['silence_ratio']
        
        # Pause length similarity
        pause_diff = abs(profile1.estimated_pause_length - profile2.estimated_pause_length)
        pause_score = max(0, 1 - pause_diff / 3.0)  # Normalize by max expected difference
        total_score += pause_score * weights['estimated_pause_length']
        
        # Dynamic range similarity
        range_diff = abs(profile1.dynamic_range - profile2.dynamic_range)
        range_score = max(0, 1 - range_diff / 40.0)  # Normalize by typical range
        total_score += range_score * weights['dynamic_range']
        
        return min(1.0, max(0.0, total_score))
    
    def find_similar_profiles(self, target_profile: AudioProfile, 
                            optimization_history: List[OptimizationResult], 
                            threshold: float = 0.8) -> List[OptimizationResult]:
        """
        Find optimization results with similar audio profiles.
        
        Args:
            target_profile: Profile to match against
            optimization_history: List of previous optimization results
            threshold: Minimum similarity score
            
        Returns:
            List of similar optimization results, sorted by similarity
        """
        similar_results = []
        
        for result in optimization_history:
            if result.audio_profile:
                similarity = self.calculate_similarity_score(target_profile, result.audio_profile)
                if similarity >= threshold:
                    similar_results.append((similarity, result))
        
        # Sort by similarity score (descending)
        similar_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the results (without similarity scores)
        return [result for _, result in similar_results]
