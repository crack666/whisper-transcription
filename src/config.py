"""
Configuration and constants for the study material processor
"""

# Language mapping for Whisper
LANGUAGE_MAP = {
    "english": "en", 
    "german": "de", 
    "french": "fr", 
    "spanish": "es", 
    "italian": "it", 
    "japanese": "ja", 
    "chinese": "zh", 
    "portuguese": "pt",
    "russian": "ru", 
    "arabic": "ar", 
    "korean": "ko", 
    "turkish": "tr",
    "hindi": "hi", 
    "dutch": "nl", 
    "swedish": "sv", 
    "indonesian": "id",
    "polish": "pl", 
    "finnish": "fi", 
    "czech": "cs", 
    "danish": "da",
    "auto": None
}

# Available Whisper models
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]

# Supported video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

# Supported audio file extensions
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

# Default configuration values
DEFAULT_CONFIG = {
    'transcription': {
        'model': 'large-v3',
        'language': 'german',
        'device': None,  # Auto-detect
        'min_silence_len': 1000,
        'padding': 750,
        'silence_adjustment': 3.0,
        'cleanup_segments': True
    },
    'screenshots': {
        'similarity_threshold': 0.85,
        'min_time_between_shots': 2.0,
        'frame_check_interval': 1.0,
        'resize_for_comparison': (320, 240)
    },
    'pdf_matching': {
        'max_preview_chars': 1000,
        'max_pages_preview': 3
    },
    'output': {
        'cleanup_audio': False,
        'generate_html': True,
        'generate_json': True
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}