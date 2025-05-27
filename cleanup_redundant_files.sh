#!/bin/bash

echo "🧹 Cleaning up redundant files in whisper-transcription..."
echo "=================================================="

# Navigate to project directory
cd "$(dirname "$0")"

echo "📍 Current directory: $(pwd)"
echo ""

# Function to safely remove file if it exists
safe_remove() {
    if [ -f "$1" ]; then
        echo "❌ Removing: $1"
        rm "$1"
    else
        echo "⏭️  Not found: $1 (already clean)"
    fi
}

# Function to safely remove directory if it exists
safe_remove_dir() {
    if [ -d "$1" ]; then
        echo "❌ Removing directory: $1"
        rm -rf "$1"
    else
        echo "⏭️  Not found: $1 (already clean)"
    fi
}

echo "🗑️  Removing obsolete test files..."
safe_remove "test_enhanced_final.py"
safe_remove "test_enhanced_fix.py"
safe_remove "test_fixed_config.py"
safe_remove "test_slow_speaker.py"

echo ""
echo "🗑️  Removing redundant tools..."
safe_remove "optimize_settings.py"
safe_remove "quick_transcribe.py"

echo ""
echo "🗑️  Removing temporary logs and results..."
safe_remove "optimization_log.txt"
safe_remove "optimization_results.json"
safe_remove "debug_transcription.log"

echo ""
echo "🗑️  Removing Node.js dependencies (not needed for Python project)..."
safe_remove "package.json"
safe_remove "package-lock.json"
safe_remove_dir "node_modules"

echo ""
echo "🎬 Test media files (keeping for now - remove manually if needed):"
if [ -f "TestFile_cut.mp4" ]; then
    size=$(du -h "TestFile_cut.mp4" | cut -f1)
    echo "   📹 TestFile_cut.mp4 ($size) - useful for testing"
fi
if [ -f "interview.mp3" ]; then
    size=$(du -h "interview.mp3" | cut -f1)
    echo "   🎙️  interview.mp3 ($size) - useful for testing"
fi

echo ""
echo "✅ CLEANUP COMPLETE!"
echo "=================================================="
echo ""
echo "💾 Important files kept:"
echo "   ✅ study_processor_v2.py (main program)"
echo "   ✅ auto_optimize.py (auto-optimization system)"
echo "   ✅ process_studies_v2.py (batch processing)"
echo "   ✅ transcription_analyzer.py (audio analysis)"
echo "   ✅ test_video_v2.py (system testing)"
echo "   ✅ test_system.sh (comprehensive testing)"
echo "   ✅ debug_transcription.py (debugging)"
echo "   ✅ optimize_audio_only.py (audio optimization)"
echo "   ✅ optimization_database.json (learning database)"
echo "   ✅ audio_optimization_results.json (optimization results)"
echo "   ✅ src/ (modular source code)"
echo "   ✅ configs/ (configuration files)"
echo "   ✅ README.md (documentation)"
echo "   ✅ requirements.txt (dependencies)"
echo ""
echo "🎯 Repository is now clean and professional!"
echo ""
echo "⚠️  To remove test media files (saves ~50MB):"
echo "   rm TestFile_cut.mp4 interview.mp3"
echo ""
echo "📊 Check remaining files:"
echo "   ls -la *.py *.json *.md *.txt"