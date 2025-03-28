# Advanced Audio Transcription Tool

A Python script for transcribing audio files using OpenAI's Whisper model with enhanced features. The tool optimizes transcription by splitting audio into segments based on silence detection, supports parallel processing, tracks progress with ETA, and offers multiple output formats including timestamps and speaker diarization.

## Features

- Choice of any Whisper model (tiny to large-v3)
- Parallel processing with multiple workers for faster transcription
- Real-time progress tracking with ETA estimates
- Optional speaker diarization (experimental)
- Multiple output formats (txt, json, srt, vtt)
- Optional timestamps in the transcription output
- Automatically detects background noise levels for optimal silence detection
- Splits audio on silent intervals to improve transcription accuracy
- Supports resuming interrupted transcription jobs
- Error handling with retries for failed segments
- Progress tracking with segment-level processing

## Requirements

- Python 3.8+
- OpenAI Whisper
- PyDub
- FFmpeg (required for audio processing)
- PyAnnote Audio (optional, for speaker diarization)

## Installation

1. Ensure FFmpeg is installed on your system
2. Install required Python packages:

```bash
pip install -r requirements.txt
```

For speaker diarization support:
```bash
pip install pyannote.audio
```
Note: Speaker diarization requires a Hugging Face token (set as HF_TOKEN environment variable)

## Usage

Basic usage:

```bash
python audio_transcription.py --audio_file=your_audio.mp3 --output_file=transcription.txt
```

With advanced features:

```bash
python audio_transcription.py --audio_file=your_audio.mp3 --output_file=transcription.txt --language=english --model=medium --workers=4 --include_timestamps --output_format=json
```

With speaker diarization:

```bash
python audio_transcription.py --audio_file=your_audio.mp3 --output_file=transcription.srt --output_format=srt --include_timestamps --diarize --max_speakers=3
```

### Examples

Conversational audio:
```bash
python audio_transcription.py --audio_file=interview.mp3 --output_file=interview.txt --min_silence_len=300 --padding=400 --workers=2 --include_timestamps --diarize
```

Lecture or presentation:
```bash
python audio_transcription.py --audio_file=lecture.mp3 --output_file=lecture.vtt --output_format=vtt --model=large-v3 --min_silence_len=1000 --padding=500 --include_timestamps
```

## Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--audio_file` | Path to the audio file to transcribe | Required | Any supported audio format (mp3, wav, etc.) |
| `--output_file` | Path to save the transcription output | Required | File path |
| `--language` | Language for transcription | "german" | english, german, french, spanish, auto, etc. |
| `--model` | Whisper model to use | "large-v3" | tiny, base, small, medium, large, large-v2, large-v3 |
| `--workers` | Number of parallel workers | 1 | 1-8 (based on CPU cores) |
| `--min_silence_len` | Minimum silence length in ms | 1000 | 200-2000 |
| `--padding` | Padding in ms added to segment boundaries | 750 | 100-1000 |
| `--silence_adjustment` | Adjustment to silence threshold in dB | 3.0 | 1.0-10.0 |
| `--output_format` | Format of the output file | "txt" | txt, json, srt, vtt |
| `--include_timestamps` | Include timestamps in output | False | Flag (no value needed) |
| `--diarize` | Enable speaker diarization | False | Flag (no value needed) |
| `--max_speakers` | Maximum number of speakers for diarization | 2 | 1-10 |
| `--no_fp16` | Disable FP16 (for CPUs without AVX2) | False | Flag (no value needed) |
| `--keep_segments` | Keep audio segments after processing | False | Flag (no value needed) |
| `--retry_attempts` | Number of retry attempts for failed segments | 3 | 1-10 |
| `--resume` | Resume transcription from existing segments | False | Flag (no value needed) |
| `--verbose` | Enable verbose output | False | Flag (no value needed) |
| `--debug` | Enable debug mode | False | Flag (no value needed) |

## Output Formats

- **txt**: Simple text format, optionally with timestamps and speaker labels
- **json**: Structured JSON format with timestamps, confidence scores, and speaker information
- **srt**: SubRip subtitle format with timestamps and speaker labels
- **vtt**: WebVTT subtitle format for web video players

## Parameter Selection Guide

- **model**: Choose based on accuracy needs and available resources:
  - tiny/base: Fast but less accurate, good for simple audio
  - small/medium: Balanced performance
  - large/large-v3: Highest accuracy but slower and more resource-intensive

- **min_silence_len**: Controls how the audio is split:
  - Lower values (200-500ms): Better for fast-paced conversations with frequent speaker changes
  - Higher values (800-1500ms): Better for lectures or speeches with distinct pauses
  - Common value: 300ms for conversational audio, 1000ms for presentations

- **workers**: Number of parallel processing threads:
  - Single-core devices: Use 1
  - Multi-core devices: Use 2-4 for optimal performance
  - High-end systems: Can use 4-8

## How It Works

1. The script analyzes background noise levels in the audio
2. Audio is split into segments based on detected silence
3. Segments are processed in parallel using the selected Whisper model
4. Optional speaker diarization identifies different speakers
5. Results are saved in the selected output format
6. Progress is tracked with real-time ETA estimates

## Advanced Scenarios

- **Low-quality audio**: Decrease silence_adjustment (1.0-2.0) and decrease min_silence_len (300-500ms)
- **Multiple speakers**: Enable diarization with --diarize and set appropriate --max_speakers
- **Video subtitles**: Use --output_format=srt or --output_format=vtt with --include_timestamps
- **Data analysis**: Use --output_format=json to get structured data with confidence scores

## Limitations

- Speaker diarization is experimental and requires a Hugging Face token
- Very large files may require significant processing time and memory
- Performance depends on the quality of the audio and background noise levels
- The tiny and base models support fewer languages than larger models