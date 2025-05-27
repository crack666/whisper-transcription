#!/usr/bin/env python3
"""
Adaptive Configuration Optimizer for Whisper Transcription.

This module automatically determines optimal transcription settings based on
audio characteristics and speaker patterns. It learns from previous optimizations
and can adapt to different audio profiles (speakers, recording conditions, etc.).
"""

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add src directory to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import extract_audio_from_video

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
    """Result from a single optimization test."""
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
        result = asdict(self)
        result['audio_profile'] = self.audio_profile.to_dict()
        return result

class AdaptiveOptimizer:
    """
    Adaptive configuration optimizer that learns optimal settings
    for different audio profiles and speaker patterns.
    """
    
    def __init__(self, optimization_db_path: str = "optimization_database.json", default_model: str = "large-v3"):
        self.db_path = optimization_db_path
        self.default_model = default_model
        self.logger = logging.getLogger(__name__)
        self.optimization_history = self.load_optimization_history()
        
        # Configuration templates for different optimization strategies
        self.base_configs = {
            "conservative": {
                "min_silence_len": 1000,
                "silence_adjustment": 5.0,
                "min_segment_length": 1000,
                "padding": 1000
            },
            "moderate": {
                "min_silence_len": 1500,
                "silence_adjustment": 4.0,
                "min_segment_length": 1500,
                "padding": 1500
            },
            "aggressive": {
                "min_silence_len": 2000,
                "silence_adjustment": 3.0,
                "min_segment_length": 800,
                "padding": 2000
            },
            "ultra_sensitive": {
                "min_silence_len": 500,
                "silence_adjustment": 8.0,
                "min_segment_length": 500,
                "padding": 750
            },
            "lecture_optimized": {
                "min_silence_len": 2500,
                "silence_adjustment": 4.0,
                "min_segment_length": 1200,
                "padding": 2000
            }
        }
    
    def analyze_audio_profile(self, audio_file: str) -> AudioProfile:
        """Analyze audio characteristics to create a profile."""
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent
            
            # Load audio
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0  # seconds
            
            # Calculate volume statistics
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            # Convert to dBFS
            rms = np.sqrt(np.mean(samples**2))
            mean_volume_db = 20 * np.log10(rms / (2**15)) if rms > 0 else -np.inf
            
            # Volume standard deviation
            volume_chunks = []
            chunk_size = len(samples) // 100  # 100 chunks
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms > 0:
                        chunk_db = 20 * np.log10(chunk_rms / (2**15))
                        volume_chunks.append(chunk_db)
            
            volume_std = np.std(volume_chunks) if volume_chunks else 0
            
            # Silence analysis
            silence_thresh = mean_volume_db + 10  # Rough threshold
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=silence_thresh)
            
            if nonsilent_ranges:
                speech_duration = sum(end - start for start, end in nonsilent_ranges) / 1000.0
                speech_ratio = speech_duration / duration
                silence_ratio = 1 - speech_ratio
                
                # Estimate pause lengths
                if len(nonsilent_ranges) > 1:
                    gaps = []
                    for i in range(len(nonsilent_ranges) - 1):
                        gap = nonsilent_ranges[i+1][0] - nonsilent_ranges[i][1]
                        gaps.append(gap)
                    estimated_pause_length = np.mean(gaps) if gaps else 1000
                else:
                    estimated_pause_length = 1000
            else:
                speech_ratio = 0.1
                silence_ratio = 0.9
                estimated_pause_length = 2000
            
            # Speaker type classification
            if speech_ratio > 0.8:
                speaker_type = "dense_speech"
            elif speech_ratio > 0.6:
                speaker_type = "moderate_speech"
            elif speech_ratio > 0.3:
                speaker_type = "sparse_speech"
            else:
                speaker_type = "very_sparse"
            
            # Background noise estimation
            silent_chunks = []
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms > 0:
                        chunk_db = 20 * np.log10(chunk_rms / (2**15))
                        silent_chunks.append(chunk_db)
            
            background_noise_level = np.percentile(silent_chunks, 10) if silent_chunks else -40
            dynamic_range = np.percentile(silent_chunks, 90) - np.percentile(silent_chunks, 10) if silent_chunks else 20
            
            # File hash for uniqueness
            with open(audio_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            
            return AudioProfile(
                duration_seconds=duration,
                mean_volume_db=mean_volume_db,
                volume_std=volume_std,
                silence_ratio=silence_ratio,
                speech_ratio=speech_ratio,
                speaker_type=speaker_type,
                estimated_pause_length=estimated_pause_length,
                background_noise_level=background_noise_level,
                dynamic_range=dynamic_range,
                file_hash=file_hash
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze audio profile: {e}")
            # Return default profile
            return AudioProfile(
                duration_seconds=150,
                mean_volume_db=-20,
                volume_std=5,
                silence_ratio=0.3,
                speech_ratio=0.7,
                speaker_type="moderate_speech",
                estimated_pause_length=1500,
                background_noise_level=-30,
                dynamic_range=20,
                file_hash="unknown"
            )
    
    def find_similar_profiles(self, target_profile: AudioProfile, threshold: float = 0.8) -> List[OptimizationResult]:
        """Find optimization results from similar audio profiles."""
        similar_results = []
        
        for result in self.optimization_history:
            if not result.success:
                continue
                
            profile = result.audio_profile
            
            # Calculate similarity score based on multiple factors
            similarity_factors = []
            
            # Speaker type match (high weight)
            if profile.speaker_type == target_profile.speaker_type:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.3)
            
            # Speech ratio similarity
            speech_diff = abs(profile.speech_ratio - target_profile.speech_ratio)
            speech_sim = max(0, 1 - speech_diff / 0.5)  # Normalize by 0.5 range
            similarity_factors.append(speech_sim)
            
            # Pause length similarity
            pause_diff = abs(profile.estimated_pause_length - target_profile.estimated_pause_length)
            pause_sim = max(0, 1 - pause_diff / 2000)  # Normalize by 2 second range
            similarity_factors.append(pause_sim)
            
            # Volume level similarity
            volume_diff = abs(profile.mean_volume_db - target_profile.mean_volume_db)
            volume_sim = max(0, 1 - volume_diff / 20)  # Normalize by 20dB range
            similarity_factors.append(volume_sim)
            
            # Dynamic range similarity
            range_diff = abs(profile.dynamic_range - target_profile.dynamic_range)
            range_sim = max(0, 1 - range_diff / 30)  # Normalize by 30dB range
            similarity_factors.append(range_sim)
            
            # Weighted average (speaker type gets double weight)
            weights = [2.0, 1.0, 1.0, 1.0, 1.0]
            overall_similarity = sum(f * w for f, w in zip(similarity_factors, weights)) / sum(weights)
            
            if overall_similarity >= threshold:
                similar_results.append(result)
        
        # Sort by similarity (higher first)
        return sorted(similar_results, key=lambda r: self.calculate_similarity_score(r.audio_profile, target_profile), reverse=True)
    
    def calculate_similarity_score(self, profile1: AudioProfile, profile2: AudioProfile) -> float:
        """Calculate detailed similarity score between two profiles."""
        if profile1.speaker_type == profile2.speaker_type:
            type_sim = 1.0
        else:
            type_sim = 0.3
        
        speech_sim = max(0, 1 - abs(profile1.speech_ratio - profile2.speech_ratio) / 0.5)
        pause_sim = max(0, 1 - abs(profile1.estimated_pause_length - profile2.estimated_pause_length) / 2000)
        volume_sim = max(0, 1 - abs(profile1.mean_volume_db - profile2.mean_volume_db) / 20)
        range_sim = max(0, 1 - abs(profile1.dynamic_range - profile2.dynamic_range) / 30)
        
        weights = [2.0, 1.0, 1.0, 1.0, 1.0]
        factors = [type_sim, speech_sim, pause_sim, volume_sim, range_sim]
        
        return sum(f * w for f, w in zip(factors, weights)) / sum(weights)
    
    def generate_optimized_configs(self, audio_profile: AudioProfile) -> List[Dict]:
        """Generate optimized configurations based on audio profile."""
        configs = []
        
        # Check for similar profiles first
        similar_results = self.find_similar_profiles(audio_profile)
        
        if similar_results:
            self.logger.info(f"Found {len(similar_results)} similar audio profiles")
            
            # Use best performing configurations from similar profiles
            best_similar = sorted(similar_results, key=lambda r: r.word_count, reverse=True)[:3]
            
            for i, result in enumerate(best_similar):
                config = result.parameters.copy()
                config_name = f"learned_optimal_{i+1}"
                configs.append({
                    "name": config_name,
                    "transcriber": "standard",  # Most results show standard works better
                    "model": self.default_model,  # Use realistic model
                    **config,
                    "source": "learned_from_similar"
                })
        
        # Generate profile-adaptive configurations
        adaptive_configs = self.generate_profile_adaptive_configs(audio_profile)
        configs.extend(adaptive_configs)
        
        # Add base configurations as fallback
        for base_name, base_config in self.base_configs.items():
            configs.append({
                "name": f"base_{base_name}",
                "transcriber": "standard",
                "model": self.default_model,
                **base_config,
                "source": "base_template"
            })
        
        return configs
    
    def generate_profile_adaptive_configs(self, profile: AudioProfile) -> List[Dict]:
        """Generate configurations specifically adapted to the audio profile."""
        configs = []
        
        # Adaptive configuration based on speech patterns
        if profile.speaker_type == "dense_speech":
            # Dense speech - need more aggressive silence detection
            configs.append({
                "name": "adaptive_dense_speech",
                "transcriber": "standard",
                "model": self.default_model,
                "min_silence_len": max(1000, int(profile.estimated_pause_length * 0.5)),
                "silence_adjustment": 3.0,
                "min_segment_length": 800,
                "padding": 1500,
                "source": "adaptive_dense"
            })
            
        elif profile.speaker_type == "sparse_speech":
            # Sparse speech - longer pauses, more conservative
            configs.append({
                "name": "adaptive_sparse_speech",
                "transcriber": "standard",
                "model": self.default_model,
                "min_silence_len": min(3000, int(profile.estimated_pause_length * 1.2)),
                "silence_adjustment": 5.0,
                "min_segment_length": 1500,
                "padding": 2000,
                "source": "adaptive_sparse"
            })
            
        else:
            # Moderate speech - balanced approach
            configs.append({
                "name": "adaptive_moderate_speech",
                "transcriber": "standard",
                "model": self.default_model,
                "min_silence_len": int(profile.estimated_pause_length),
                "silence_adjustment": 4.0,
                "min_segment_length": 1200,
                "padding": 1500,
                "source": "adaptive_moderate"
            })
        
        # Volume-adaptive configuration
        if profile.mean_volume_db < -25:
            # Quiet audio - more sensitive detection
            silence_adj = 8.0
        elif profile.mean_volume_db > -15:
            # Loud audio - less sensitive
            silence_adj = 2.0
        else:
            # Normal volume
            silence_adj = 4.0
        
        configs.append({
            "name": "adaptive_volume_optimized",
            "transcriber": "standard",
            "model":  self.default_model,
            "min_silence_len": 1500,
            "silence_adjustment": silence_adj,
            "min_segment_length": 1000,
            "padding": 1500,
            "source": "adaptive_volume"
        })
        
        return configs
    
    def run_adaptive_optimization(self, audio_file: str, max_configs: int = 8) -> Dict:
        """Run adaptive optimization on an audio file."""
        self.logger.info(f"🎯 Starting adaptive optimization for: {audio_file}")
        
        # Analyze audio profile
        self.logger.info("📊 Analyzing audio profile...")
        audio_profile = self.analyze_audio_profile(audio_file)
        
        self.logger.info(f"   Speaker type: {audio_profile.speaker_type}")
        self.logger.info(f"   Speech ratio: {audio_profile.speech_ratio:.2f}")
        self.logger.info(f"   Estimated pause length: {audio_profile.estimated_pause_length:.0f}ms")
        self.logger.info(f"   Mean volume: {audio_profile.mean_volume_db:.1f}dB")
        
        # Generate optimized configurations
        self.logger.info("🧠 Generating adaptive configurations...")
        test_configs = self.generate_optimized_configs(audio_profile)[:max_configs]
        
        self.logger.info(f"   Generated {len(test_configs)} configurations")
        for config in test_configs:
            self.logger.info(f"   - {config['name']} ({config.get('source', 'unknown')})")
        
        # Run optimization tests
        results = []
        for i, config in enumerate(test_configs, 1):
            self.logger.info(f"\n--- Test {i}/{len(test_configs)}: {config['name']} ---")
            
            try:
                result = self.test_configuration(config, audio_file, audio_profile)
                results.append(result)
                
                self.logger.info(f"   ✅ {result.word_count} words, {result.segment_count} segments, {result.processing_time:.1f}s")
                
            except Exception as e:
                self.logger.error(f"   ❌ Test failed: {e}")
                
            time.sleep(1)  # Brief pause between tests
        
        # Analyze results and save to history
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            # Add to optimization history
            self.optimization_history.extend(successful_results)
            self.save_optimization_history()
            
            # Find best configuration
            best_result = max(successful_results, key=lambda r: r.word_count)
            
            analysis = {
                "audio_profile": audio_profile.to_dict(),
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "best_config": {
                    "name": best_result.config_name,
                    "word_count": best_result.word_count,
                    "parameters": best_result.parameters,
                    "processing_time": best_result.processing_time
                },
                "all_results": [r.to_dict() for r in results]
            }
            
            self.logger.info(f"\n🏆 Best configuration: {best_result.config_name}")
            self.logger.info(f"   Words: {best_result.word_count}")
            self.logger.info(f"   Time: {best_result.processing_time:.1f}s")
            
            return analysis
        else:
            self.logger.error("❌ No successful configurations found")
            return {
                "audio_profile": audio_profile.to_dict(),
                "total_tests": len(results),
                "successful_tests": 0,
                "error": "No configurations succeeded"
            }
    
    def test_configuration(self, config: Dict, audio_file: str, audio_profile: AudioProfile) -> OptimizationResult:
        """Test a single configuration."""
        from src.transcriber import AudioTranscriber
        
        start_time = time.time()
        
        # Create transcriber with config
        transcriber_config = {
            'min_silence_len': config['min_silence_len'],
            'padding': config['padding'],
            'silence_adjustment': config['silence_adjustment'],
            'cleanup_segments': True
        }
        
        transcriber = AudioTranscriber(
            model_name=config['model'],
            language='german',
            config=transcriber_config
        )
        
        # Run transcription
        result = transcriber.transcribe_audio_file(audio_file)
        
        # Extract metrics
        transcript = result.get('full_text', '')
        segments = result.get('segments', [])
        word_count = len(transcript.split()) if transcript else 0
        char_count = len(transcript) if transcript else 0
        segment_count = len(segments)
        
        processing_time = time.time() - start_time
        
        # Calculate quality score (simple heuristic)
        quality_score = min(1.0, word_count / 100.0)  # Normalize to 1.0 at 100 words
        
        return OptimizationResult(
            config_name=config['name'],
            parameters=config,
            word_count=word_count,
            char_count=char_count,
            segment_count=segment_count,
            processing_time=processing_time,
            success=word_count > 0,
            quality_score=quality_score,
            audio_profile=audio_profile,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def create_optimized_config_file(self, best_result: OptimizationResult, output_path: str):
        """Create an optimized configuration file based on results."""
        config = {
            "transcription": {
                "model": best_result.parameters.get("model", "large-v3"),
                "language": "german",
                "device": None,
                "min_silence_len": best_result.parameters["min_silence_len"],
                "padding": best_result.parameters["padding"],
                "silence_adjustment": best_result.parameters["silence_adjustment"],
                "max_segment_length": 300000,
                "min_segment_length": best_result.parameters.get("min_segment_length", 1000),
                "overlap_duration": best_result.parameters.get("overlap_duration", 1000),
                "enable_vad": False,
                "normalize_audio": False,
                "use_beam_search": True,
                "best_of": 3,
                "patience": 2.0,
                "length_penalty": 1.0,
                "merge_short_segments": True,
                "validate_transcription": True,
                "cleanup_segments": True
            },
            "screenshots": {
                "similarity_threshold": 0.80,
                "min_time_between_shots": 3.0,
                "frame_check_interval": 1.0,
                "resize_for_comparison": [320, 240]
            },
            "pdf_matching": {
                "max_preview_chars": 1500,
                "max_pages_preview": 5
            },
            "output": {
                "extract_screenshots": True,
                "generate_html": True,
                "generate_json": True,
                "cleanup_audio": False
            },
            "use_enhanced_transcriber": False,
            "optimization_metadata": {
                "audio_profile": best_result.audio_profile.to_dict(),
                "optimization_result": {
                    "word_count": best_result.word_count,
                    "processing_time": best_result.processing_time,
                    "timestamp": best_result.timestamp
                },
                "description": f"Auto-optimized for {best_result.audio_profile.speaker_type} with {best_result.word_count} words"
            },
            "description": f"Auto-optimized configuration - {best_result.word_count} words in {best_result.processing_time:.1f}s"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Optimized config saved to: {output_path}")
    
    def load_optimization_history(self) -> List[OptimizationResult]:
        """Load optimization history from database."""
        if not os.path.exists(self.db_path):
            return []
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history = []
            for item in data:
                try:
                    # Convert audio_profile dict back to AudioProfile object
                    profile_data = item['audio_profile']
                    audio_profile = AudioProfile.from_dict(profile_data)
                    
                    # Create OptimizationResult
                    result = OptimizationResult(
                        config_name=item['config_name'],
                        parameters=item['parameters'],
                        word_count=item['word_count'],
                        char_count=item['char_count'],
                        segment_count=item['segment_count'],
                        processing_time=item['processing_time'],
                        success=item['success'],
                        quality_score=item['quality_score'],
                        audio_profile=audio_profile,
                        timestamp=item['timestamp']
                    )
                    history.append(result)
                except KeyError as e:
                    self.logger.warning(f"Skipping malformed history entry: {e}")
                    continue
            
            self.logger.info(f"📚 Loaded {len(history)} optimization results from history")
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization history: {e}")
            return []
    
    def save_optimization_history(self):
        """Save optimization history to database."""
        try:
            data = [result.to_dict() for result in self.optimization_history]
            
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved {len(self.optimization_history)} optimization results")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization history: {e}")

def setup_logging():
    """Setup logging for adaptive optimizer."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function for testing the adaptive optimizer."""
    setup_logging()
    
    test_file = "TestFile_cut.mp4"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"🚀 Starting adaptive optimization")
    print(f"📁 Test file: {test_file}")
    
    optimizer = AdaptiveOptimizer()
    
    # Run adaptive optimization
    analysis = optimizer.run_adaptive_optimization(test_file)
    
    if "best_config" in analysis:
        # Create optimized config file
        best_result = None
        for result in optimizer.optimization_history:
            if (result.config_name == analysis["best_config"]["name"] and 
                result.word_count == analysis["best_config"]["word_count"]):
                best_result = result
                break
        
        if best_result:
            optimizer.create_optimized_config_file(
                best_result, 
                f"configs/auto_optimized_{int(time.time())}.json"
            )
    
    # Save analysis
    with open("adaptive_optimization_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Adaptive optimization completed!")
    print(f"📊 Analysis saved to: adaptive_optimization_analysis.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())