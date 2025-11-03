#!/usr/bin/env python3
"""
Configuration Generator for Whisper Transcription Optimization.

This module generates optimized transcription configurations based on audio profiles,
historical data, and intelligent parameter variations.
"""

import logging
from typing import Dict, List, Optional
from .audio_profile_analyzer import AudioProfile, OptimizationResult

class ConfigurationGenerator:
    """Generates optimized transcription configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_optimized_configs(self, audio_profile: AudioProfile, 
                                 optimization_history: List[OptimizationResult]) -> List[Dict]:
        """
        Generate optimized configurations based on audio profile and history.
        
        Args:
            audio_profile: Analyzed audio characteristics
            optimization_history: Previous optimization results
            
        Returns:
            List of optimized configuration dictionaries
        """
        configs = []
        
        # Add profile-adaptive configurations
        profile_configs = self.generate_profile_adaptive_configs(audio_profile)
        configs.extend(profile_configs)
        
        # Add configurations based on historical best performers
        if optimization_history:
            historical_configs = self._generate_historical_configs(optimization_history, audio_profile)
            configs.extend(historical_configs)
        
        # Add baseline configurations
        baseline_configs = self._generate_baseline_configs()
        configs.extend(baseline_configs)
        
        # Remove duplicates and limit count
        unique_configs = self._remove_duplicate_configs(configs)
        
        self.logger.info(f"Generated {len(unique_configs)} unique configurations")
        return unique_configs[:12]  # Limit to reasonable number
    
    def generate_profile_adaptive_configs(self, profile: AudioProfile) -> List[Dict]:
        """
        Generate configurations specifically adapted to audio profile characteristics.
        
        Args:
            profile: Audio profile to adapt to
            
        Returns:
            List of adapted configuration dictionaries
        """
        configs = []
        
        if profile.speaker_type == "very_sparse":
            # Very conservative approach for very sparse content
            configs.extend([
                {
                    "name": "very_sparse_conservative",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 3500,
                    "silence_adjustment": 2.0,
                    "min_segment_length": 2500,
                    "padding": 3000,
                    "source": "profile_adaptive"
                },
                {
                    "name": "very_sparse_balanced",
                    "transcriber": "standard", 
                    "model": "large-v3",
                    "min_silence_len": 2800,
                    "silence_adjustment": 3.5,
                    "min_segment_length": 2000,
                    "padding": 2500,
                    "source": "profile_adaptive"
                }
            ])
        
        elif profile.speaker_type == "sparse":
            # Optimized for sparse speech content
            configs.extend([
                {
                    "name": "sparse_optimized",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 2200,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 1500,
                    "padding": 2000,
                    "source": "profile_adaptive"
                },
                {
                    "name": "sparse_sensitive",
                    "transcriber": "standard",
                    "model": "large-v3", 
                    "min_silence_len": 1800,
                    "silence_adjustment": 5.0,
                    "min_segment_length": 1200,
                    "padding": 1800,
                    "source": "profile_adaptive"
                }
            ])
        
        elif profile.speaker_type == "slow":
            # Adapted for slow speakers with long pauses
            configs.extend([
                {
                    "name": "slow_speaker_optimized",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 2500,
                    "silence_adjustment": 3.0,
                    "min_segment_length": 2000,
                    "padding": 2500,
                    "source": "profile_adaptive"
                },
                {
                    "name": "slow_speaker_patient",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 3000,
                    "silence_adjustment": 2.5,
                    "min_segment_length": 2500,
                    "padding": 3000,
                    "source": "profile_adaptive"
                }
            ])
        
        elif profile.speaker_type == "expressive":
            # Adapted for expressive speakers with dynamic range
            configs.extend([
                {
                    "name": "expressive_adaptive",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 1200,
                    "silence_adjustment": 6.0,
                    "min_segment_length": 1000,
                    "padding": 1500,
                    "source": "profile_adaptive"
                },
                {
                    "name": "expressive_balanced",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 1500,
                    "silence_adjustment": 4.5,
                    "min_segment_length": 1200,
                    "padding": 1800,
                    "source": "profile_adaptive"
                }
            ])
        
        else:  # Normal speaker
            configs.extend([
                {
                    "name": "normal_optimized",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 1500,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 1200,
                    "padding": 1500,
                    "source": "profile_adaptive"
                },
                {
                    "name": "normal_sensitive",
                    "transcriber": "standard",
                    "model": "large-v3",
                    "min_silence_len": 1200,
                    "silence_adjustment": 5.0,
                    "min_segment_length": 1000,
                    "padding": 1200,
                    "source": "profile_adaptive"
                }
            ])
        
        # Add configurations based on specific audio characteristics
        if profile.silence_ratio > 0.6:  # Very quiet audio
            configs.append({
                "name": "high_silence_adapted",
                "transcriber": "standard",
                "model": "large-v3",
                "min_silence_len": int(2000 + profile.estimated_pause_length * 500),
                "silence_adjustment": 3.0,
                "min_segment_length": 1500,
                "padding": 2000,
                "source": "silence_adapted"
            })
        
        if profile.background_noise_level > -35:  # Noisy environment
            configs.append({
                "name": "noise_resistant",
                "transcriber": "standard",
                "model": "large-v3",
                "min_silence_len": 1800,
                "silence_adjustment": 6.0,  # Higher sensitivity for noisy audio
                "min_segment_length": 1000,
                "padding": 1500,
                "source": "noise_adapted"
            })
        
        return configs
    
    def generate_intelligent_parameter_variations(self, base_config: Dict, 
                                                audio_profile: AudioProfile,
                                                optimization_history: List[OptimizationResult],
                                                count: int = 6) -> List[Dict]:
        """
        Generate intelligent parameter variations based on analysis and audio profile.
        
        Args:
            base_config: Base configuration to vary
            audio_profile: Audio characteristics
            optimization_history: Historical optimization data
            count: Number of variations to generate
            
        Returns:
            List of parameter variation configurations
        """
        variations = []
        
        # Get parameter insights from historical data
        param_analysis = self._analyze_parameter_impact(optimization_history)
        
        # Base proven configuration
        proven_best = self._find_best_performing_config(optimization_history)
        if proven_best:
            base_variant = {
                "name": "proven_best_config",
                "transcriber": "standard",
                "model": base_config['model'],
                "min_silence_len": proven_best['min_silence_len'],
                "silence_adjustment": proven_best['silence_adjustment'],
                "min_segment_length": proven_best.get('min_segment_length', 1000),
                "padding": proven_best.get('padding', 1000),
                "source": "proven_best"
            }
            variations.append(base_variant)
        
        # Audio profile adapted configurations
        if audio_profile.speaker_type == "very_sparse":
            variations.extend([
                {
                    "name": "very_sparse_optimized",
                    "transcriber": "standard", 
                    "model": base_config['model'],
                    "min_silence_len": 3000,
                    "silence_adjustment": 3.0,
                    "min_segment_length": 2000,
                    "padding": 2500,
                    "source": "very_sparse_optimized"
                },
                {
                    "name": "very_sparse_enhanced",
                    "transcriber": "standard",
                    "model": base_config['model'], 
                    "min_silence_len": 2500,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 1500,
                    "padding": 2000,
                    "source": "very_sparse_enhanced"
                }
            ])
        
        elif audio_profile.speaker_type == "sparse":
            variations.extend([
                {
                    "name": "sparse_balanced",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 2000,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 1200,
                    "padding": 1500,
                    "source": "sparse_balanced"
                },
                {
                    "name": "sparse_sensitive",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1200,
                    "silence_adjustment": 5.5,
                    "min_segment_length": 1000,
                    "padding": 1200,
                    "source": "sparse_sensitive"
                }
            ])
        
        # Parameter analysis based variations
        if param_analysis and "error" not in param_analysis:
            # Best silence length range
            silence_analysis = param_analysis.get("min_silence_len_analysis", {})
            if silence_analysis.get("best_range") == "short":
                variations.append({
                    "name": "optimized_variant_1",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1000,
                    "silence_adjustment": 5.0,
                    "min_segment_length": 800,
                    "padding": 1000,
                    "source": "intelligent_variation"
                })
            
            # Best adjustment range
            adj_analysis = param_analysis.get("silence_adjustment_analysis", {})
            if adj_analysis.get("best_range") == "high":
                variations.append({
                    "name": "optimized_variant_2", 
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1200,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 800,
                    "padding": 1200,
                    "source": "intelligent_variation"
                })
        
        # Conservative fallback configurations
        variations.extend([
            {
                "name": "conservative_baseline", 
                "transcriber": "standard",
                "model": base_config['model'],
                "min_silence_len": 1500,
                "silence_adjustment": 3.5,
                "min_segment_length": 1000,
                "padding": 1500,
                "source": "conservative_baseline"
            },
            {
                "name": "medium_sensitivity",
                "transcriber": "standard", 
                "model": base_config['model'],
                "min_silence_len": 1200,
                "silence_adjustment": 4.5,
                "min_segment_length": 900,
                "padding": 1200,
                "source": "medium_sensitivity"
            }
        ])
        
        # Remove duplicates and limit count
        unique_variations = self._remove_duplicate_configs(variations)
        return unique_variations[:count]
    
    def _generate_historical_configs(self, optimization_history: List[OptimizationResult], 
                                   audio_profile: AudioProfile) -> List[Dict]:
        """Generate configurations based on historical best performers."""
        configs = []
        
        # Find top performing configurations
        top_results = sorted(optimization_history, key=lambda x: x.word_count, reverse=True)[:3]
        
        for i, result in enumerate(top_results):
            config = {
                "name": f"historical_best_{i+1}",
                "transcriber": "standard",
                "model": "large-v3",
                "min_silence_len": result.parameters.get('min_silence_len', 1500),
                "silence_adjustment": result.parameters.get('silence_adjustment', 4.0),
                "min_segment_length": result.parameters.get('min_segment_length', 1000),
                "padding": result.parameters.get('padding', 1500),
                "source": "historical_best"
            }
            configs.append(config)
        
        return configs
    
    def _generate_baseline_configs(self) -> List[Dict]:
        """Generate baseline configurations as fallbacks."""
        return [
            {
                "name": "conservative_baseline",
                "transcriber": "standard",
                "model": "large-v3",
                "min_silence_len": 1800,
                "silence_adjustment": 3.0,
                "min_segment_length": 1200,
                "padding": 1800,
                "source": "baseline"
            },
            {
                "name": "balanced_baseline",
                "transcriber": "standard",
                "model": "large-v3", 
                "min_silence_len": 1500,
                "silence_adjustment": 4.0,
                "min_segment_length": 1000,
                "padding": 1500,
                "source": "baseline"
            },
            {
                "name": "aggressive_baseline",
                "transcriber": "standard",
                "model": "large-v3",
                "min_silence_len": 1200,
                "silence_adjustment": 5.0,
                "min_segment_length": 800,
                "padding": 1200,
                "source": "baseline"
            }
        ]
    
    def _remove_duplicate_configs(self, configs: List[Dict]) -> List[Dict]:
        """Remove duplicate configurations based on key parameters."""
        seen_signatures = set()
        unique_configs = []
        
        for config in configs:
            # Create signature based on key parameters
            signature = (
                config.get('min_silence_len', 0),
                config.get('silence_adjustment', 0),
                config.get('padding', 0)
            )
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_configs.append(config)
        
        return unique_configs
    
    def _analyze_parameter_impact(self, optimization_history: List[OptimizationResult]) -> Dict:
        """Analyze parameter impact from historical data."""
        if not optimization_history:
            return {"error": "No optimization history available"}
        
        # Group results by parameter ranges
        silence_len_groups = {"short": [], "medium": [], "long": []}
        adjustment_groups = {"low": [], "medium": [], "high": []}
        
        for result in optimization_history:
            params = result.parameters
            silence_len = params.get('min_silence_len', 1500)
            adjustment = params.get('silence_adjustment', 4.0)
            
            # Categorize silence length
            if silence_len < 1300:
                silence_len_groups["short"].append(result)
            elif silence_len < 2000:
                silence_len_groups["medium"].append(result)
            else:
                silence_len_groups["long"].append(result)
            
            # Categorize adjustment
            if adjustment < 3.5:
                adjustment_groups["low"].append(result)
            elif adjustment < 5.0:
                adjustment_groups["medium"].append(result)
            else:
                adjustment_groups["high"].append(result)
        
        # Calculate averages
        def calc_avg_words(group):
            return sum(r.word_count for r in group) / len(group) if group else 0
        
        silence_analysis = {
            "short_avg": calc_avg_words(silence_len_groups["short"]),
            "medium_avg": calc_avg_words(silence_len_groups["medium"]),
            "long_avg": calc_avg_words(silence_len_groups["long"])
        }
        
        adjustment_analysis = {
            "low_avg": calc_avg_words(adjustment_groups["low"]),
            "medium_avg": calc_avg_words(adjustment_groups["medium"]),
            "high_avg": calc_avg_words(adjustment_groups["high"])
        }
        
        # Find best ranges
        best_silence_range = max(silence_analysis.keys(), key=lambda k: silence_analysis[k])
        best_adjustment_range = max(adjustment_analysis.keys(), key=lambda k: adjustment_analysis[k])
        
        return {
            "min_silence_len_analysis": {
                "best_range": best_silence_range,
                "best_range_avg_words": silence_analysis[best_silence_range],
                "recommendation": f"Use {best_silence_range} silence length values"
            },
            "silence_adjustment_analysis": {
                "best_range": best_adjustment_range,
                "best_range_avg_words": adjustment_analysis[best_adjustment_range],
                "recommendation": f"Use {best_adjustment_range} adjustment values"
            }
        }
    
    def _find_best_performing_config(self, optimization_history: List[OptimizationResult]) -> Optional[Dict]:
        """Find the best performing configuration from history."""
        if not optimization_history:
            return None
        
        best_result = max(optimization_history, key=lambda x: x.word_count)
        return best_result.parameters
