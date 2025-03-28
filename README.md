# Audio Transcription Tool

A Python script for transcribing audio files using OpenAI's Whisper model with silence-based segmentation. The tool optimizes transcription by splitting audio into segments based on silence detection. The tool optimizes transcription by splitting audio into segments based on silence detection.

## Features

- Automatically detects background noise levels for optimal silence detection
- Splits audio on silent intervals to improve transcription accuracy
- Supports resuming interrupted transcription jobs
- Error handling with retries for failed segments
- Progress tracking with segment-level processing

## Requirements

- Python 3.6+
- OpenAI Whisper
- PyDub
- FFmpeg (required for audio processing)

## Installation

1. Ensure FFmpeg is installed on your system
2. Install required Python packages:

```bash
pip install openai-whisper pydub
```

## Usage

Basic usage:

```bash
python audio_transcription.py --audio_file=your_audio.mp3 --output_file=transcription.txt
```

With custom parameters:

```bash
python audio_transcription.py --audio_file=your_audio.mp3 --output_file=transcription.txt --language=en --min_silence_len=500 --padding=300
```

### Example

```bash
python audio_transcription.py --audio_file=2025-02-11_13-40-20_2.mp3 --output_file=Intension4_11_02_2025.txt --min_silence_len=300 --padding=400
```

## Parameters

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--audio_file` | Path to the audio file to transcribe | Required | Any supported audio format (mp3, wav, etc.) |
| `--output_file` | Path to save the transcription output | Required | Text file path |
| `--language` | Language code for transcription | "German" | Language code (en, de, fr, etc.) |
| `--min_silence_len` | Minimum silence length in ms | 1000 | 300-1000 (shorter for conversational audio) |
| `--padding` | Padding in ms added to start/end of segments | 750 | 300-750 (adjust based on speech pace) |
| `--resume` | Resume transcription from existing segments | False | Flag (no value needed) |

## Parameter Selection Guide

- **min_silence_len**: Controls how the audio is split:
  - Lower values (200-500ms): Better for fast-paced conversations or when speakers frequently interrupt each other
  - Higher values (800-1500ms): Better for lectures or speeches with distinct pauses
  - Common value: 300ms for conversational audio, 1000ms for presentations

- **padding**: Ensures context is maintained between segments:
  - Lower values (100-300ms): Minimal overlap between segments
  - Higher values (400-750ms): More context preserved at segment boundaries
  - Common value: 400ms for most applications

## How It Works

1. The script analyzes background noise levels in the audio
2. Audio is split into segments based on detected silence
3. Each segment is transcribed using Whisper's large-v3 model
4. Successfully transcribed segments are removed to save space
5. Results are combined into a single output file

## Limitations

- Very large files may require significant processing time
- Performance depends on the quality of the audio and background noise levels