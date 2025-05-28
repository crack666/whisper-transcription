#!/usr/bin/env python3
"""
Analyze Overlapping Segments for Duplicate Words
Check if our fixed-time segmentation with overlap causes duplicate word detection.
"""

import os
import sys
import logging
import time
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.enhanced_transcriber import EnhancedAudioTranscriber

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def analyze_segment_overlaps(segments):
    """Analyze segments for potential overlapping content."""
    
    print(f"\n🔍 ANALYZING SEGMENT OVERLAPS")
    print(f"=" * 50)
    
    overlaps_detected = []
    
    for i in range(len(segments) - 1):
        current_segment = segments[i]
        next_segment = segments[i + 1]
        
        # Skip segments with errors
        if current_segment.get('error') or next_segment.get('error'):
            continue
            
        current_text = current_segment.get('text', '').strip()
        next_text = next_segment.get('text', '').strip()
        
        if not current_text or not next_text:
            continue
        
        # Check timing overlap
        current_end = current_segment.get('end_time', 0)
        next_start = next_segment.get('start_time', 0)
        
        # Check for text similarity in overlap region
        current_words = current_text.split()
        next_words = next_text.split()
        
        # Look for similar endings and beginnings
        similarity_detected = False
        overlap_words = []
        
        # Check last N words of current vs first N words of next
        for n in range(1, min(10, len(current_words), len(next_words)) + 1):
            current_ending = " ".join(current_words[-n:])
            next_beginning = " ".join(next_words[:n])
            
            # Calculate similarity
            similarity = SequenceMatcher(None, current_ending.lower(), next_beginning.lower()).ratio()
            
            if similarity > 0.8:  # 80% similar
                similarity_detected = True
                overlap_words = current_words[-n:]
                break
        
        if similarity_detected or (current_end > next_start):
            overlap_info = {
                "segment_pair": (i, i+1),
                "timing_overlap": current_end - next_start if current_end > next_start else 0,
                "text_similarity": similarity_detected,
                "overlap_words": overlap_words,
                "current_end_words": current_words[-5:] if len(current_words) >= 5 else current_words,
                "next_start_words": next_words[:5] if len(next_words) >= 5 else next_words,
                "current_segment": current_segment.get('segment_id', i),
                "next_segment": next_segment.get('segment_id', i+1)
            }
            overlaps_detected.append(overlap_info)
            
            print(f"⚠️  Overlap detected between segments {i} and {i+1}:")
            print(f"   Timing overlap: {overlap_info['timing_overlap']/1000:.1f}s")
            print(f"   Text similarity: {overlap_info['text_similarity']}")
            if overlap_words:
                print(f"   Duplicate words: {' '.join(overlap_words)}")
            print(f"   Current ending: '...{' '.join(overlap_info['current_end_words'])}'")
            print(f"   Next beginning: '{' '.join(overlap_info['next_start_words'])}...'")
    
    return overlaps_detected

def count_word_duplicates(full_text):
    """Count duplicate words in the full text."""
    
    print(f"\n📊 WORD DUPLICATE ANALYSIS")
    print(f"=" * 50)
    
    # Clean and normalize text
    words = re.findall(r'\b\w+\b', full_text.lower())
    word_counts = Counter(words)
    
    # Find words that appear multiple times
    duplicates = {word: count for word, count in word_counts.items() if count > 1}
    
    total_words = len(words)
    unique_words = len(word_counts)
    duplicate_instances = sum(count - 1 for count in duplicates.values())
    
    print(f"📈 Text Statistics:")
    print(f"   Total words: {total_words}")
    print(f"   Unique words: {unique_words}")
    print(f"   Duplicate instances: {duplicate_instances}")
    print(f"   Duplication rate: {(duplicate_instances/total_words)*100:.1f}%")
    
    # Show most common duplicates (excluding common words)
    common_words = {'der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'mit', 'von', 'auf', 'für', 'als', 'bei', 'nach', 'über', 'durch', 'um', 'an', 'oder', 'auch', 'nicht', 'ein', 'eine', 'einen', 'einem', 'einer', 'hat', 'haben', 'wird', 'werden', 'kann', 'könnte', 'soll', 'sollte', 'aber', 'wenn', 'dann', 'weil', 'dass', 'sich', 'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'uns', 'mich', 'dir', 'ihm', 'ihn', 'sie'}
    
    content_duplicates = {word: count for word, count in duplicates.items() 
                         if word not in common_words and len(word) > 3}
    
    if content_duplicates:
        print(f"\n🔍 Suspicious content word duplicates:")
        sorted_duplicates = sorted(content_duplicates.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_duplicates[:10]:  # Top 10
            print(f"   '{word}': {count} times")
    else:
        print(f"✅ No suspicious content word duplicates found")
    
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "duplicate_instances": duplicate_instances,
        "duplication_rate": (duplicate_instances/total_words)*100,
        "content_duplicates": content_duplicates
    }

def test_overlap_deduplication(audio_file: str):
    """Test different overlap settings and analyze duplication."""
    
    print(f"\n🎯 TESTING OVERLAP DEDUPLICATION")
    print(f"=" * 60)
    print(f"Audio file: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    # Test different overlap settings
    test_configs = [
        {
            "name": "No Overlap",
            "config": {
                "segmentation_mode": "fixed_time",
                "fixed_time_duration": 25000,  # 25 seconds
                "fixed_time_overlap": 0,       # No overlap
                "cleanup_segments": True
            }
        },
        {
            "name": "Small Overlap (1s)",
            "config": {
                "segmentation_mode": "fixed_time",
                "fixed_time_duration": 25000,  # 25 seconds
                "fixed_time_overlap": 1000,    # 1 second overlap
                "cleanup_segments": True
            }
        },
        {
            "name": "Medium Overlap (2s)",
            "config": {
                "segmentation_mode": "fixed_time",
                "fixed_time_duration": 25000,  # 25 seconds
                "fixed_time_overlap": 2000,    # 2 seconds overlap
                "cleanup_segments": True
            }
        },
        {
            "name": "Large Overlap (3s)",
            "config": {
                "segmentation_mode": "fixed_time",
                "fixed_time_duration": 25000,  # 25 seconds
                "fixed_time_overlap": 3000,    # 3 seconds overlap
                "cleanup_segments": True
            }
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n🔧 Testing {test_config['name']}...")
        
        try:
            start_time = time.time()
            
            # Create transcriber with test configuration
            transcriber = EnhancedAudioTranscriber(
                model_name="tiny",  # Fast model for testing
                language="german",
                config=test_config['config']
            )
            
            # Run transcription
            result = transcriber.transcribe_audio_file_enhanced(audio_file)
            
            processing_time = time.time() - start_time
            
            # Extract data
            segments = result.get('segments', [])
            full_text = result.get('full_text', '')
            
            if not full_text:
                print(f"   ❌ No text generated")
                continue
            
            # Analyze overlaps
            overlaps = analyze_segment_overlaps(segments)
            
            # Analyze word duplicates
            duplicate_stats = count_word_duplicates(full_text)
            
            test_result = {
                "name": test_config['name'],
                "overlap_ms": test_config['config']['fixed_time_overlap'],
                "segments_count": len(segments),
                "words_count": duplicate_stats['total_words'],
                "unique_words": duplicate_stats['unique_words'],
                "duplication_rate": duplicate_stats['duplication_rate'],
                "segment_overlaps": len(overlaps),
                "processing_time": processing_time,
                "content_duplicates": len(duplicate_stats['content_duplicates']),
                "suspicious_duplicates": list(duplicate_stats['content_duplicates'].keys())[:5]
            }
            
            print(f"   ✅ Results:")
            print(f"      Words: {test_result['words_count']} (unique: {test_result['unique_words']})")
            print(f"      Duplication rate: {test_result['duplication_rate']:.1f}%")
            print(f"      Segment overlaps detected: {test_result['segment_overlaps']}")
            print(f"      Content word duplicates: {test_result['content_duplicates']}")
            
            results.append(test_result)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary comparison
    if results:
        print(f"\n📊 OVERLAP DUPLICATION COMPARISON")
        print(f"=" * 80)
        print(f"{'Config':<20} {'Overlap':<8} {'Words':<8} {'Unique':<8} {'Dup%':<8} {'SegOvlp':<8} {'ContDup':<8}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for result in results:
            print(f"{result['name']:<20} {result['overlap_ms']/1000:.1f}s    {result['words_count']:<8} "
                  f"{result['unique_words']:<8} {result['duplication_rate']:<8.1f} "
                  f"{result['segment_overlaps']:<8} {result['content_duplicates']:<8}")
    
    return results

def main():
    """Main analysis function."""
    setup_logging()
    
    # Test with the shorter interview file
    audio_file = "interview_cut.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ Test file not found: {audio_file}")
        print("Please ensure interview_cut.mp3 exists in the current directory")
        return 1
    
    try:
        results = test_overlap_deduplication(audio_file)
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print(f"=" * 50)
        
        if results:
            # Find optimal overlap setting
            sorted_by_duplicates = sorted(results, key=lambda x: x['duplication_rate'])
            best_result = sorted_by_duplicates[0]
            
            print(f"✅ Optimal overlap setting: {best_result['name']}")
            print(f"   Overlap: {best_result['overlap_ms']/1000:.1f}s")
            print(f"   Duplication rate: {best_result['duplication_rate']:.1f}%")
            print(f"   Total words: {best_result['words_count']}")
            
            if best_result['duplication_rate'] > 5:  # More than 5% duplication
                print(f"\n⚠️  High duplication detected!")
                print(f"   Consider:")
                print(f"   - Reducing overlap duration")
                print(f"   - Adding deduplication post-processing")
                print(f"   - Using adaptive overlap based on speech density")
            else:
                print(f"\n✅ Duplication rate is acceptable")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
