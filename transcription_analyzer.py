#!/usr/bin/env python3
"""
Transcription Quality Analyzer - Tool to analyze and improve transcription results
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.enhanced_transcriber import EnhancedAudioTranscriber
from src.transcriber import AudioTranscriber
from pydub import AudioSegment

def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze transcription quality and suggest improvements")
    
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to audio file to analyze")
    parser.add_argument("--output", type=str, default="./analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--compare", action="store_true",
                       help="Compare standard vs enhanced transcriber")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--model", type=str, default="base",
                       help="Whisper model for quick analysis (default: base)")
    parser.add_argument("--language", type=str, default="german",
                       help="Language for transcription")
    
    return parser

def analyze_audio_properties(audio_file: str) -> Dict:
    """Analyze basic audio properties."""
    print(f"📊 Analyzing audio properties: {Path(audio_file).name}")
    
    audio = AudioSegment.from_file(audio_file)
    
    # Basic properties
    duration = len(audio) / 1000.0
    sample_rate = audio.frame_rate
    channels = audio.channels
    
    # Volume analysis
    chunk_size = 1000  # 1 second
    volumes = []
    timestamps = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) > 0:
            volumes.append(chunk.dBFS)
            timestamps.append(i / 1000.0)
    
    volumes = np.array(volumes)
    
    # Statistics
    properties = {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "mean_volume_db": np.mean(volumes),
        "volume_std": np.std(volumes),
        "min_volume_db": np.min(volumes),
        "max_volume_db": np.max(volumes),
        "dynamic_range": np.max(volumes) - np.min(volumes),
        "volume_timeline": list(zip(timestamps, volumes.tolist()))
    }
    
    # Detect silence patterns
    silence_threshold = properties["mean_volume_db"] - properties["volume_std"]
    quiet_periods = volumes < silence_threshold
    
    # Find continuous quiet periods
    quiet_durations = []
    in_quiet = False
    quiet_start = 0
    
    for i, is_quiet in enumerate(quiet_periods):
        if is_quiet and not in_quiet:
            quiet_start = i
            in_quiet = True
        elif not is_quiet and in_quiet:
            quiet_duration = (i - quiet_start) * (chunk_size / 1000.0)
            quiet_durations.append(quiet_duration)
            in_quiet = False
    
    properties.update({
        "silence_threshold_db": silence_threshold,
        "quiet_ratio": np.sum(quiet_periods) / len(quiet_periods),
        "num_quiet_periods": len(quiet_durations),
        "avg_quiet_duration": np.mean(quiet_durations) if quiet_durations else 0,
        "max_quiet_duration": np.max(quiet_durations) if quiet_durations else 0,
        "quiet_durations": quiet_durations
    })
    
    print(f"   Duration: {duration:.1f}s")
    print(f"   Mean volume: {properties['mean_volume_db']:.1f} dBFS")
    print(f"   Quiet ratio: {properties['quiet_ratio']:.1%}")
    print(f"   Max quiet period: {properties['max_quiet_duration']:.1f}s")
    
    return properties

def compare_transcribers(audio_file: str, model: str, language: str) -> Dict:
    """Compare standard vs enhanced transcriber."""
    print(f"\n🔀 Comparing transcribers...")
    
    results = {}
    
    # Standard transcriber
    print("   Testing standard transcriber...")
    standard_transcriber = AudioTranscriber(model_name=model, language=language)
    standard_result = standard_transcriber.transcribe_audio_file(audio_file)
    
    results["standard"] = {
        "segments_count": len(standard_result.get("segments", [])),
        "total_text_length": len(standard_result.get("full_text", "")),
        "processing_time": standard_result.get("processing_time_seconds", 0),
        "success_rate": (standard_result.get("segments_successful", 0) / 
                        max(standard_result.get("segments_total", 1), 1)),
        "word_count": len(standard_result.get("full_text", "").split()),
        "segments": standard_result.get("segments", [])
    }
    
    # Enhanced transcriber
    print("   Testing enhanced transcriber...")
    enhanced_transcriber = EnhancedAudioTranscriber(model_name=model, language=language)
    enhanced_result = enhanced_transcriber.transcribe_audio_file_enhanced(audio_file)
    
    results["enhanced"] = {
        "segments_count": len(enhanced_result.get("segments", [])),
        "total_text_length": len(enhanced_result.get("full_text", "")),
        "processing_time": enhanced_result.get("processing_time_seconds", 0),
        "success_rate": (enhanced_result.get("segments_successful", 0) / 
                        max(enhanced_result.get("segments_total", 1), 1)),
        "word_count": len(enhanced_result.get("full_text", "").split()),
        "average_quality": enhanced_result.get("average_quality_score", 0),
        "speech_analysis": enhanced_result.get("speech_analysis", {}),
        "segments": enhanced_result.get("segments", [])
    }
    
    # Calculate improvements
    results["improvement"] = {
        "segments_change": results["enhanced"]["segments_count"] - results["standard"]["segments_count"],
        "text_length_change": results["enhanced"]["total_text_length"] - results["standard"]["total_text_length"],
        "word_count_change": results["enhanced"]["word_count"] - results["standard"]["word_count"],
        "time_change": results["enhanced"]["processing_time"] - results["standard"]["processing_time"]
    }
    
    print(f"   Standard: {results['standard']['word_count']} words in {results['standard']['segments_count']} segments")
    print(f"   Enhanced: {results['enhanced']['word_count']} words in {results['enhanced']['segments_count']} segments")
    print(f"   Improvement: +{results['improvement']['word_count_change']} words (+{results['improvement']['segments_change']} segments)")
    
    return results

def analyze_segment_coverage(segments: List[Dict], total_duration: float) -> Dict:
    """Analyze how well segments cover the total audio duration."""
    if not segments:
        return {"coverage_ratio": 0, "gaps": [], "overlaps": []}
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x.get("start_time", 0))
    
    # Calculate covered time
    covered_time = 0
    gaps = []
    overlaps = []
    
    prev_end = 0
    
    for segment in sorted_segments:
        start = segment.get("start_time", 0) / 1000.0  # Convert to seconds
        end = segment.get("end_time", 0) / 1000.0
        
        # Check for gap
        if start > prev_end:
            gap_duration = start - prev_end
            if gap_duration > 1.0:  # Only report gaps > 1 second
                gaps.append({
                    "start": prev_end,
                    "end": start,
                    "duration": gap_duration
                })
        
        # Check for overlap
        elif start < prev_end and prev_end > 0:
            overlap_duration = prev_end - start
            overlaps.append({
                "start": start,
                "end": prev_end,
                "duration": overlap_duration
            })
        
        covered_time += end - start
        prev_end = end
    
    # Check final gap
    if prev_end < total_duration:
        final_gap = total_duration - prev_end
        if final_gap > 1.0:
            gaps.append({
                "start": prev_end,
                "end": total_duration,
                "duration": final_gap
            })
    
    coverage_ratio = covered_time / total_duration if total_duration > 0 else 0
    
    return {
        "coverage_ratio": coverage_ratio,
        "covered_time": covered_time,
        "total_gaps": len(gaps),
        "total_gap_time": sum(g["duration"] for g in gaps),
        "total_overlaps": len(overlaps),
        "total_overlap_time": sum(o["duration"] for o in overlaps),
        "gaps": gaps,
        "overlaps": overlaps
    }

def generate_recommendations(audio_props: Dict, coverage_analysis: Dict, 
                           comparison: Optional[Dict] = None) -> List[str]:
    """Generate recommendations for improving transcription."""
    recommendations = []
    
    # Based on audio properties
    if audio_props["quiet_ratio"] > 0.4:
        recommendations.append(
            f"📢 High quiet ratio ({audio_props['quiet_ratio']:.1%}): "
            "Use min_silence_len=3000-4000ms for better pause detection"
        )
    
    if audio_props["max_quiet_duration"] > 10:
        recommendations.append(
            f"⏸️ Very long pauses detected (max: {audio_props['max_quiet_duration']:.1f}s): "
            "Use enhanced transcriber with overlap segments"
        )
    
    if audio_props["dynamic_range"] > 30:
        recommendations.append(
            f"🔊 High dynamic range ({audio_props['dynamic_range']:.1f}dB): "
            "Consider audio normalization before transcription"
        )
    
    # Based on coverage analysis
    if coverage_analysis["coverage_ratio"] < 0.8:
        recommendations.append(
            f"⚠️ Low coverage ratio ({coverage_analysis['coverage_ratio']:.1%}): "
            "Increase padding to 2000ms and reduce silence threshold"
        )
    
    if coverage_analysis["total_gaps"] > 5:
        recommendations.append(
            f"🕳️ Many gaps detected ({coverage_analysis['total_gaps']}): "
            "Use enhanced transcriber with overlapping segments"
        )
    
    if coverage_analysis["total_gap_time"] > 30:
        recommendations.append(
            f"⏰ Significant time lost in gaps ({coverage_analysis['total_gap_time']:.1f}s): "
            "Use more aggressive silence detection parameters"
        )
    
    # Based on comparison (if available)
    if comparison and comparison["improvement"]["word_count_change"] > 50:
        recommendations.append(
            "✅ Enhanced transcriber shows significant improvement: "
            "Use enhanced transcriber for production"
        )
    
    # General recommendations
    if not recommendations:
        recommendations.append("✅ Audio quality looks good for standard transcription")
    
    return recommendations

def create_visualizations(audio_props: Dict, coverage_analysis: Dict, 
                         comparison: Optional[Dict], output_dir: str):
    """Create visualization plots."""
    if not audio_props.get("volume_timeline"):
        print("⚠️ No volume data for visualization")
        return
    
    print("📈 Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Volume timeline plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Volume over time
    timestamps, volumes = zip(*audio_props["volume_timeline"])
    axes[0].plot(timestamps, volumes, linewidth=1, alpha=0.7)
    axes[0].axhline(y=audio_props["silence_threshold_db"], color='r', linestyle='--', 
                   label=f"Silence threshold ({audio_props['silence_threshold_db']:.1f} dBFS)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Volume (dBFS)")
    axes[0].set_title("Audio Volume Timeline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Coverage visualization
    if coverage_analysis["gaps"]:
        # Show gaps in coverage
        for gap in coverage_analysis["gaps"]:
            axes[1].axvspan(gap["start"], gap["end"], alpha=0.3, color='red', 
                           label='Gap' if gap == coverage_analysis["gaps"][0] else "")
    
    # Show covered periods (simplified)
    covered_start = 0
    for gap in coverage_analysis["gaps"]:
        if gap["start"] > covered_start:
            axes[1].axvspan(covered_start, gap["start"], alpha=0.3, color='green',
                           label='Covered' if covered_start == 0 else "")
        covered_start = gap["end"]
    
    # Final covered period
    if covered_start < audio_props["duration_seconds"]:
        axes[1].axvspan(covered_start, audio_props["duration_seconds"], 
                       alpha=0.3, color='green')
    
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Coverage")
    axes[1].set_title(f"Transcription Coverage ({coverage_analysis['coverage_ratio']:.1%})")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "audio_analysis.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path / 'audio_analysis.png'}")
    
    # Comparison plot (if available)
    if comparison:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        categories = ['Segments', 'Words', 'Text Length']
        standard_values = [
            comparison["standard"]["segments_count"],
            comparison["standard"]["word_count"],
            comparison["standard"]["total_text_length"]
        ]
        enhanced_values = [
            comparison["enhanced"]["segments_count"],
            comparison["enhanced"]["word_count"],
            comparison["enhanced"]["total_text_length"]
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, standard_values, width, label='Standard', alpha=0.7)
        ax.bar(x + width/2, enhanced_values, width, label='Enhanced', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Count')
        ax.set_title('Standard vs Enhanced Transcriber Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "transcriber_comparison.png", dpi=150, bbox_inches='tight')
        print(f"   Saved: {output_path / 'transcriber_comparison.png'}")
    
    plt.close('all')

def main():
    """Main analysis function."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)
    
    print(f"🎙️ Transcription Quality Analyzer")
    print(f"Audio file: {args.audio}")
    print(f"Output: {args.output}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Analyze audio properties
        audio_props = analyze_audio_properties(args.audio)
        
        # Compare transcribers if requested
        comparison = None
        if args.compare:
            comparison = compare_transcribers(args.audio, args.model, args.language)
        
        # Analyze coverage (use enhanced transcriber for detailed analysis)
        print(f"\n📋 Analyzing transcription coverage...")
        enhanced_transcriber = EnhancedAudioTranscriber(model_name=args.model, language=args.language)
        result = enhanced_transcriber.transcribe_audio_file_enhanced(args.audio)
        
        coverage_analysis = analyze_segment_coverage(
            result.get("segments", []), 
            audio_props["duration_seconds"]
        )
        
        print(f"   Coverage: {coverage_analysis['coverage_ratio']:.1%}")
        print(f"   Gaps: {coverage_analysis['total_gaps']} ({coverage_analysis['total_gap_time']:.1f}s)")
        
        # Generate recommendations
        print(f"\n💡 Recommendations:")
        recommendations = generate_recommendations(audio_props, coverage_analysis, comparison)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Create visualizations
        if args.visualize:
            try:
                create_visualizations(audio_props, coverage_analysis, comparison, args.output)
            except ImportError:
                print("⚠️ Matplotlib not available, skipping visualizations")
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
        
        # Save detailed analysis
        analysis_data = {
            "audio_file": args.audio,
            "audio_properties": audio_props,
            "coverage_analysis": coverage_analysis,
            "recommendations": recommendations,
            "comparison": comparison,
            "transcription_result": {
                "segments_count": len(result.get("segments", [])),
                "word_count": len(result.get("full_text", "").split()),
                "average_quality": result.get("average_quality_score", 0),
                "speech_analysis": result.get("speech_analysis", {})
            }
        }
        
        analysis_file = output_dir / "detailed_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Audio duration: {audio_props['duration_seconds']:.1f}s")
        print(f"   Transcription coverage: {coverage_analysis['coverage_ratio']:.1%}")
        print(f"   Words transcribed: {len(result.get('full_text', '').split())}")
        print(f"   Average quality score: {result.get('average_quality_score', 0):.2f}")
        
        if comparison:
            improvement = comparison["improvement"]["word_count_change"]
            print(f"   Enhanced vs Standard: +{improvement} words ({improvement/max(comparison['standard']['word_count'], 1)*100:+.1f}%)")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()