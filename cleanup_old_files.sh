#!/bin/bash
# cleanup_old_files.sh - Remove obsolete v1.0 files after refactoring

echo "🧹 Cleaning up obsolete v1.0 files..."

# Function to safely remove file if it exists
safe_remove() {
    if [ -f "$1" ]; then
        echo "   Removing: $1"
        rm "$1"
    fi
}

# Function to safely remove directory if it exists
safe_remove_dir() {
    if [ -d "$1" ]; then
        echo "   Removing directory: $1"
        rm -rf "$1"
    fi
}

echo "📄 Removing old scripts..."
safe_remove "audio_transcription.py"
safe_remove "enhanced_transcription.py"
safe_remove "study_material_processor.py"
safe_remove "process_studies.py"
safe_remove "test_single_video.py"

echo "📚 Removing old documentation..."
safe_remove "ENHANCED_README.md"
safe_remove "MODULAR_README.md"
safe_remove "README_old.md"

echo "🗂️ Removing backup/temporary files..."
safe_remove "transcript.txt"
safe_remove "transcript.txt.bak"

echo "🎵 Removing temporary audio files from studies/..."
find studies/ -name "*.wav" -type f -delete 2>/dev/null || true

echo "📦 Checking Node.js files..."
if [ -f "package.json" ]; then
    echo "   ⚠️  Found Node.js files (package.json, package-lock.json, node_modules/)"
    echo "   ❓ Remove Node.js files? (y/N): "
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        safe_remove "package.json"
        safe_remove "package-lock.json"
        safe_remove_dir "node_modules"
        echo "   ✅ Node.js files removed"
    else
        echo "   ⏭️  Keeping Node.js files"
    fi
fi

echo "🔧 Checking git helper scripts..."
if [ -f "push-whisper-repo.sh" ]; then
    echo "   ❓ Remove git helper scripts? (y/N): "
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        safe_remove "push-whisper-repo.sh"
        safe_remove "setup-github-auth.sh"
        echo "   ✅ Git helper scripts removed"
    else
        echo "   ⏭️  Keeping git helper scripts"
    fi
fi

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📁 Remaining files structure:"
echo "📍 Main entry points:"
echo "   study_processor_v2.py          # Main program"
echo "   process_studies_v2.py          # Convenience script"
echo "   test_video_v2.py               # Single video test"
echo "   test_slow_speaker.py           # Transcription optimization test"
echo "   transcription_analyzer.py      # Quality analysis tool"
echo ""
echo "📍 Configuration:"
echo "   configs/slow_speaker.json      # For slow speakers"
echo "   configs/lecture_optimized.json # For very slow speakers"
echo ""
echo "📍 Documentation:"
echo "   README.md                       # Main documentation"
echo "   TRANSCRIPTION_IMPROVEMENTS.md  # Transcription details"
echo "   CLEANUP_GUIDE.md               # This cleanup guide"
echo ""
echo "📍 Modules:"
echo "   src/                           # All modular components"
echo ""
echo "🚀 Ready to use v2.0! Try:"
echo "   python study_processor_v2.py --validate"
echo "   python process_studies_v2.py"