#!/usr/bin/env python3
"""
Systematic optimization of transcription settings using word count as metric.
Tests multiple configurations to find optimal parameters for lecture transcription.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup logging for optimization tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization_log.txt'),
            logging.StreamHandler()
        ]
    )

class TranscriptionOptimizer:
    """Optimize transcription settings using word count as primary metric."""
    
    def __init__(self, test_file: str):
        self.test_file = test_file
        self.results = []
        self.logger = logging.getLogger(__name__)
        
        # Test configurations to evaluate
        self.test_configs = [
            # Standard configurations
            {
                "name": "conservative",
                "model": "tiny",
                "min_silence_len": 1000,
                "silence_adjustment": 5.0,
                "min_segment_length": 1000,
                "overlap_duration": 500
            },
            {
                "name": "moderate", 
                "model": "tiny",
                "min_silence_len": 1500,
                "silence_adjustment": 4.0,
                "min_segment_length": 1500,
                "overlap_duration": 750
            },
            {
                "name": "aggressive",
                "model": "tiny", 
                "min_silence_len": 2000,
                "silence_adjustment": 3.0,
                "min_segment_length": 2000,
                "overlap_duration": 1000
            },
            # Lecture-optimized variations
            {
                "name": "lecture_short_pauses",
                "model": "tiny",
                "min_silence_len": 800,
                "silence_adjustment": 6.0,
                "min_segment_length": 1200,
                "overlap_duration": 400
            },
            {
                "name": "lecture_medium_pauses",
                "model": "tiny",
                "min_silence_len": 1200,
                "silence_adjustment": 5.0,
                "min_segment_length": 1800,
                "overlap_duration": 600
            },
            {
                "name": "lecture_long_pauses",
                "model": "tiny",
                "min_silence_len": 2500,
                "silence_adjustment": 4.0,
                "min_segment_length": 3000,
                "overlap_duration": 1000
            },
            # Very aggressive settings
            {
                "name": "ultra_sensitive",
                "model": "tiny",
                "min_silence_len": 500,
                "silence_adjustment": 8.0,
                "min_segment_length": 800,
                "overlap_duration": 300
            },
            # Enhanced transcriber settings
            {
                "name": "enhanced_optimized",
                "model": "tiny",
                "min_silence_len": 2000,
                "silence_adjustment": 4.0,
                "min_segment_length": 2000,
                "overlap_duration": 1000,
                "use_enhanced": True
            },
            {
                "name": "enhanced_aggressive",
                "model": "tiny",
                "min_silence_len": 1500,
                "silence_adjustment": 5.0,
                "min_segment_length": 1500,
                "overlap_duration": 750,
                "use_enhanced": True
            }
        ]
    
    def create_test_config(self, params: Dict) -> str:
        """Create a temporary config file for testing."""
        config = {
            "transcription": {
                "model": params["model"],
                "language": "german",
                "device": None,
                "min_silence_len": params["min_silence_len"],
                "padding": 2000,
                "silence_adjustment": params["silence_adjustment"],
                "max_segment_length": 300000,
                "min_segment_length": params["min_segment_length"],
                "overlap_duration": params["overlap_duration"],
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
                "extract_screenshots": False,  # Disable for speed
                "generate_html": False,
                "generate_json": True,
                "cleanup_audio": True
            },
            "use_enhanced_transcriber": params.get("use_enhanced", False),
            "description": f"Test config: {params['name']}"
        }
        
        config_file = f"configs/test_{params['name']}.json"
        os.makedirs("configs", exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config_file
    
    def test_configuration(self, params: Dict) -> Dict[str, Any]:
        """Test a single configuration and return results."""
        self.logger.info(f"🧪 Testing configuration: {params['name']}")
        
        # Create test config
        config_file = self.create_test_config(params)
        
        start_time = time.time()
        
        try:
            # Import here to avoid issues if dependencies not available
            from src.processor import StudyMaterialProcessor
            
            # Create processor
            processor = StudyMaterialProcessor(config_file)
            
            # Process the test file
            result = processor.process_video(self.test_file)
            
            # Extract metrics
            transcript = result.get('transcript', '')
            segments = result.get('segments', [])
            word_count = len(transcript.split()) if transcript else 0
            char_count = len(transcript) if transcript else 0
            segment_count = len(segments)
            
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            avg_segment_length = sum(len(seg.get('text', '').split()) for seg in segments) / max(segment_count, 1)
            
            test_result = {
                "config_name": params['name'],
                "parameters": params,
                "word_count": word_count,
                "char_count": char_count,
                "segment_count": segment_count,
                "avg_segment_length": avg_segment_length,
                "processing_time": processing_time,
                "success": word_count > 0,
                "transcript_preview": transcript[:200] if transcript else "No text",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.logger.info(f"   ✅ Success: {word_count} words, {segment_count} segments, {processing_time:.1f}s")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"   ❌ Failed: {e}")
            
            return {
                "config_name": params['name'],
                "parameters": params,
                "word_count": 0,
                "char_count": 0,
                "segment_count": 0,
                "avg_segment_length": 0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        finally:
            # Cleanup
            if os.path.exists(config_file):
                os.remove(config_file)
    
    def run_optimization(self) -> List[Dict]:
        """Run optimization tests on all configurations."""
        self.logger.info(f"🎯 Starting optimization with test file: {self.test_file}")
        self.logger.info(f"📊 Testing {len(self.test_configs)} configurations")
        
        for i, config in enumerate(self.test_configs, 1):
            self.logger.info(f"\n--- Test {i}/{len(self.test_configs)} ---")
            result = self.test_configuration(config)
            self.results.append(result)
            
            # Short pause between tests
            time.sleep(2)
        
        return self.results
    
    def analyze_results(self) -> Dict:
        """Analyze results and find optimal settings."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Sort by word count (primary metric)
        successful_results = [r for r in self.results if r['success']]
        successful_results.sort(key=lambda x: x['word_count'], reverse=True)
        
        # Find best configurations
        best_overall = successful_results[0] if successful_results else None
        best_speed = min(successful_results, key=lambda x: x['processing_time']) if successful_results else None
        
        # Calculate statistics
        word_counts = [r['word_count'] for r in successful_results]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        analysis = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "failed_tests": len(self.results) - len(successful_results),
            "best_word_count": {
                "config": best_overall['config_name'] if best_overall else None,
                "words": best_overall['word_count'] if best_overall else 0,
                "parameters": best_overall['parameters'] if best_overall else None
            },
            "fastest_config": {
                "config": best_speed['config_name'] if best_speed else None,
                "time": best_speed['processing_time'] if best_speed else 0,
                "words": best_speed['word_count'] if best_speed else 0
            },
            "average_words": avg_words,
            "word_count_range": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0
            }
        }
        
        return analysis
    
    def save_results(self, filename: str = "optimization_results.json"):
        """Save results to file."""
        analysis = self.analyze_results()
        
        output = {
            "test_file": self.test_file,
            "analysis": analysis,
            "detailed_results": self.results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Results saved to: {filename}")
        return analysis
    
    def print_summary(self):
        """Print a summary of optimization results."""
        analysis = self.analyze_results()
        
        print(f"\n🎯 OPTIMIZATION SUMMARY")
        print(f"=" * 50)
        print(f"Test file: {self.test_file}")
        print(f"Total tests: {analysis['total_tests']}")
        print(f"Successful: {analysis['successful_tests']}")
        print(f"Failed: {analysis['failed_tests']}")
        
        if analysis['successful_tests'] > 0:
            print(f"\n🏆 BEST CONFIGURATION (Most Words):")
            best = analysis['best_word_count']
            print(f"   Config: {best['config']}")
            print(f"   Words: {best['words']}")
            print(f"   Parameters: {best['parameters']}")
            
            print(f"\n⚡ FASTEST CONFIGURATION:")
            fastest = analysis['fastest_config']
            print(f"   Config: {fastest['config']}")
            print(f"   Time: {fastest['time']:.1f}s")
            print(f"   Words: {fastest['words']}")
            
            print(f"\n📊 STATISTICS:")
            print(f"   Average words: {analysis['average_words']:.1f}")
            print(f"   Word range: {analysis['word_count_range']['min']} - {analysis['word_count_range']['max']}")
            
            print(f"\n📝 TOP 5 CONFIGURATIONS:")
            successful_results = [r for r in self.results if r['success']]
            successful_results.sort(key=lambda x: x['word_count'], reverse=True)
            
            for i, result in enumerate(successful_results[:5], 1):
                print(f"   {i}. {result['config_name']}: {result['word_count']} words "
                      f"({result['segment_count']} segments, {result['processing_time']:.1f}s)")

def main():
    """Main function."""
    setup_logging()
    
    test_file = "TestFile_cut.mp4"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"🚀 Starting transcription optimization")
    print(f"📁 Test file: {test_file} ({os.path.getsize(test_file)/1024/1024:.1f} MB)")
    
    optimizer = TranscriptionOptimizer(test_file)
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Save and analyze results
    analysis = optimizer.save_results()
    optimizer.print_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())