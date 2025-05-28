#!/usr/bin/env python3
"""
Integration test for the refactored AdaptiveOptimizer system.
"""
import sys
import os
sys.path.append('.')

from src.adaptive_optimizer import AdaptiveOptimizer

def test_adaptive_optimizer_integration():
    """Test the complete integration of the refactored system."""
    print("🧪 Testing AdaptiveOptimizer Integration...")
    
    # Test initialization
    try:
        optimizer = AdaptiveOptimizer()
        print("✓ AdaptiveOptimizer initialized successfully")
        print(f"  - Audio Analyzer: {type(optimizer.audio_analyzer).__name__}")
        print(f"  - Config Generator: {type(optimizer.config_generator).__name__}")
        print(f"  - Process Manager: {type(optimizer.process_manager).__name__}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test database loading
    try:
        history_count = len(optimizer.optimization_history)
        print(f"✓ Optimization history loaded: {history_count} entries")
    except Exception as e:
        print(f"❌ History loading failed: {e}")
        return False
    
    # Test component availability
    try:
        # Test audio analyzer methods
        analyzer_methods = ['analyze_audio_profile', 'calculate_similarity_score', 'find_similar_profiles']
        for method in analyzer_methods:
            if hasattr(optimizer.audio_analyzer, method):
                print(f"✓ AudioProfileAnalyzer.{method} available")
            else:
                print(f"❌ AudioProfileAnalyzer.{method} missing")
                return False
        
        # Test config generator methods
        generator_methods = ['generate_optimized_configs', 'generate_profile_adaptive_configs']
        for method in generator_methods:
            if hasattr(optimizer.config_generator, method):
                print(f"✓ ConfigurationGenerator.{method} available")
            else:
                print(f"❌ ConfigurationGenerator.{method} missing")
                return False
        
        # Test process manager methods
        manager_methods = ['test_configurations_parallel']
        for method in manager_methods:
            if hasattr(optimizer.process_manager, method):
                print(f"✓ ProcessIsolationManager.{method} available")
            else:
                print(f"❌ ProcessIsolationManager.{method} missing")
                return False
                
    except Exception as e:
        print(f"❌ Component method check failed: {e}")
        return False
    
    # Test public interface methods
    try:
        public_methods = ['run_adaptive_optimization', 'analyze_parameter_impact', 'create_optimized_config_file']
        for method in public_methods:
            if hasattr(optimizer, method):
                print(f"✓ AdaptiveOptimizer.{method} available")
            else:
                print(f"❌ AdaptiveOptimizer.{method} missing")
                return False
    except Exception as e:
        print(f"❌ Public interface check failed: {e}")
        return False
    
    print("🎉 All integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_adaptive_optimizer_integration()
    sys.exit(0 if success else 1)
