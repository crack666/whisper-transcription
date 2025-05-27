# Cleanup Guide - Veraltete Dateien entfernen

Nach dem Refactoring zu v2.0 sind einige Dateien veraltet und können entfernt werden.

## 🗑️ Dateien die entfernt werden können:

### Veraltete Scripts (v1.0)
```bash
# Diese Dateien sind durch study_processor_v2.py ersetzt
rm audio_transcription.py                    # Alte Transkription
rm enhanced_transcription.py                 # Alte erweiterte Version  
rm study_material_processor.py              # Alte Hauptversion
rm process_studies.py                       # Alte Convenience-Version
rm test_single_video.py                     # Alter Test

# Veraltete READMEs
rm ENHANCED_README.md                        # Alte Dokumentation
rm MODULAR_README.md                         # Wird in neue README integriert
```

### Node.js Relikte (falls nicht verwendet)
```bash
# Falls du Node.js nicht verwendest
rm package.json
rm package-lock.json
rm -rf node_modules/
```

### Git-Hilfsskripts (optional)
```bash
# Nur entfernen wenn nicht benötigt
rm push-whisper-repo.sh
rm setup-github-auth.sh
```

### Temporäre/Test-Dateien
```bash
# Backup-Dateien
rm transcript.txt
rm transcript.txt.bak

# Temporäre Audio-Dateien in studies/ (nach Verarbeitung)
rm studies/*.wav
```

## ✅ Dateien die BEHALTEN werden sollen:

### Aktuelle Scripts (v2.0)
- `study_processor_v2.py` - **Haupteinstiegspunkt**
- `process_studies_v2.py` - Convenience-Script für alle Videos
- `test_video_v2.py` - Test mit einem Video
- `test_slow_speaker.py` - Spezialtest für langsame Sprecher
- `transcription_analyzer.py` - Qualitätsanalyse-Tool

### Modulstruktur
- `src/` - Gesamter Modulordner
- `configs/` - Konfigurationsdateien

### Dokumentation
- `README.md` - Hauptdokumentation (wird aktualisiert)
- `TRANSCRIPTION_IMPROVEMENTS.md` - Spezielle Transkriptionsverbesserungen

### Daten
- `studies/` - Videodateien und PDFs
- `requirements.txt` - Python-Abhängigkeiten

## 🧹 Cleanup-Script

```bash
#!/bin/bash
# cleanup_old_files.sh

echo "🧹 Cleaning up old v1.0 files..."

# Remove old scripts
rm -f audio_transcription.py
rm -f enhanced_transcription.py  
rm -f study_material_processor.py
rm -f process_studies.py
rm -f test_single_video.py

# Remove old documentation
rm -f ENHANCED_README.md
rm -f MODULAR_README.md

# Remove backup files
rm -f transcript.txt
rm -f transcript.txt.bak

# Remove temporary audio files
find studies/ -name "*.wav" -type f -delete

# Optional: Remove Node.js files (uncomment if not needed)
# rm -f package.json package-lock.json
# rm -rf node_modules/

echo "✅ Cleanup completed!"
echo "📁 Remaining structure:"
ls -la
```