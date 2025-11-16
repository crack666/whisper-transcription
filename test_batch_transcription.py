"""
PROOF OF CONCEPT: TRUE Batch Transcription
==========================================

Ziel: 3 Segments gleichzeitig, 1 Model, 1 GPU

Whisper's transcribe() kann nur 1 Audio, ABER:
Wir können die internen Methoden nutzen um echtes Batching zu machen!
"""

import whisper
import torch
import numpy as np
from pathlib import Path
import time

def load_audio_batch(audio_files, sr=16000):
    """Load multiple audio files and pad to same length"""
    audios = []
    max_length = 0
    
    # Load all audios
    for file in audio_files:
        audio = whisper.load_audio(file)
        audios.append(audio)
        max_length = max(max_length, len(audio))
    
    # Pad to same length
    padded = []
    for audio in audios:
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        padded.append(audio)
    
    return np.stack(padded)  # Shape: (batch_size, audio_length)


def batch_transcribe(model, audio_files, batch_size=3):
    """
    Transcribe multiple audio files in batches using ONE model instance.
    
    This is TRUE batching: 
    - 1 model loaded (10GB VRAM)
    - Multiple audios processed together
    - GPU parallelism within single inference
    """
    results = []
    
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        print(f"📦 Processing batch {i//batch_size + 1}: {len(batch_files)} files")
        
        # Load audio batch
        audio_batch = load_audio_batch(batch_files)
        
        # Convert to mel spectrograms (batchable!)
        mel_batch = torch.stack([
            whisper.log_mel_spectrogram(torch.from_numpy(audio)).to(model.device)
            for audio in audio_batch
        ])
        print(f"   Mel batch shape: {mel_batch.shape}")  # (batch, 80, frames)
        
        # ENCODER: Process batch in parallel! 🚀
        with torch.no_grad():
            audio_features_batch = model.encoder(mel_batch)
        print(f"   Encoded features: {audio_features_batch.shape}")
        
        # DECODER: Sequential (one by one) - autoregressive limitation
        for j, audio_features in enumerate(audio_features_batch):
            # Decode uses model's decode method (not batchable easily)
            decode_result = model.decode(audio_features.unsqueeze(0))
            
            # Get text from tokens
            text = whisper.tokenizer.get_tokenizer(
                model.is_multilingual,
                language='de'
            ).decode(decode_result[0].tokens)
            
            results.append({
                'file': batch_files[j],
                'text': text
            })
    
    return results


def benchmark_batching():
    """Compare sequential vs batch processing"""
    
    # Create synthetic audio for testing (5 seconds each)
    print("🎵 Creating test audio segments...")
    sr = 16000
    duration = 5  # seconds
    num_segments = 9
    
    test_audios = []
    for i in range(num_segments):
        # Generate simple sine wave (different frequency per segment)
        t = np.linspace(0, duration, sr * duration)
        frequency = 440 + i * 50  # A4 + increments
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.3
        test_audios.append(audio)
    
    print(f"✅ Created {num_segments} test segments ({duration}s each)\n")
    
    # Load model ONCE
    print("📥 Loading model...")
    model = whisper.load_model("base", device="cuda")
    print(f"✅ Model loaded: {model.device}\n")
    
    # Sequential (current approach)
    print("=" * 60)
    print("🐌 SEQUENTIAL: 1 by 1")
    print("=" * 60)
    start = time.time()
    for i, audio in enumerate(test_audios):
        # Whisper expects 16kHz float32
        result = model.transcribe(audio, language='de')
        print(f"  ✓ Segment {i+1:02d} transcribed")
    sequential_time = time.time() - start
    print(f"⏱️  Sequential time: {sequential_time:.2f}s\n")
    
    # Batch (our approach) - using audio arrays directly
    print("=" * 60)
    print("🚀 BATCH: 3 at a time (same model)")
    print("=" * 60)
    start = time.time()
    # Process in batches
    for i in range(0, len(test_audios), 3):
        batch = test_audios[i:i+3]
        print(f"📦 Processing batch {i//3 + 1}: {len(batch)} segments")
        
        # Pad to same length
        max_len = max(len(a) for a in batch)
        padded = [np.pad(a, (0, max_len - len(a))) for a in batch]
        
        # Convert to mel spectrograms
        mel_batch = torch.stack([
            whisper.log_mel_spectrogram(torch.from_numpy(audio)).to(model.device)
            for audio in padded
        ])
        
        # Encode in batch (PARALLEL!)
        with torch.no_grad():
            audio_features_batch = model.encoder(mel_batch)
        print(f"   Encoded {len(batch)} segments in parallel")
        
        # Decode each (sequential - autoregressive)
        for j, features in enumerate(audio_features_batch):
            decode_result = model.decode(features.unsqueeze(0))
            print(f"   ✓ Decoded segment {i+j+1:02d}")
    
    batch_time = time.time() - start
    print(f"⏱️  Batch time: {batch_time:.2f}s\n")
    
    # Results
    print("=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Batch:      {batch_time:.2f}s")
    print(f"Speedup:    {sequential_time/batch_time:.2f}x")
    print(f"Improvement: {(1 - batch_time/sequential_time)*100:.1f}%")
    
    if batch_time < sequential_time:
        print("\n✅ BATCH is FASTER!")
    else:
        print("\n⚠️  Batch has overhead - need more segments for benefit")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║  TRUE BATCH TRANSCRIPTION - Proof of Concept             ║
║                                                           ║
║  Ziel: 1 Model, 3 Segments gleichzeitig                  ║
║  VRAM: ~10GB (wie sequential)                             ║
║  GPU: Besser ausgelastet durch Encoder-Batching          ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    try:
        benchmark_batching()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
