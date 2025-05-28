# 🎓 Study Material Processor v2.0

**Intelligentes System** zur automatischen Verarbeitung von Vorlesungsvideos und Audio-Dateien mit KI-basierter Transkription, Auto-Optimierung und Screenshot-Extraktion.

## ⚡ Wichtigste Features

- 🧠 **Auto-Optimierung** - Findet automatisch die besten Einstellungen für jeden Sprecher
- 🎯 **Hohe Erkennungsrate** - Bis zu 172+ Wörter pro Minute für deutsche Vorlesungen
- 📊 **Intelligente Anpassung** - Erkennt Sprechstile automatisch (langsam/schnell/Pausen)
- 📹 **Vollständige Verarbeitung** - Audio + Video + Screenshots + HTML-Reports
- 🔄 **Batch-Verarbeitung** - Automatische Verarbeitung ganzer Ordner

---

## 🚀 Einfacher Start (3 Schritte)

### 1. 🎯 Für neue Sprecher/Module (EMPFOHLEN)
```bash
# Automatische Optimierung - findet beste Einstellungen
python auto_optimize.py --input your_lecture.mp4
```

### 2. 📚 Standard-Verarbeitung  
```bash
# Vollständige Verarbeitung mit optimalen Einstellungen
python study_processor_v2.py --input your_lecture.mp4 --output ./results
```

### 3. 🔄 Weitere Videos mit gleichen Einstellungen
```bash
# Nutze die auto-generierte Konfiguration für weitere Videos
python study_processor_v2.py --input weitere_videos/ --batch --config configs/auto_optimized_*.json
```

**Das war's! 🎉** Das System erstellt automatisch optimierte Transkriptionen, Screenshots und HTML-Reports.

---

## 🛠️ Installation

```bash
# 1. Python-Abhängigkeiten installieren
pip install -r requirements.txt

# 2. FFmpeg installieren (falls nicht vorhanden)
# Windows: Download von https://ffmpeg.org/
# Ubuntu: sudo apt install ffmpeg  
# macOS: brew install ffmpeg

# 3. Setup testen
python study_processor_v2.py --validate
```

---

## 📚 Hauptanwendungsfälle

### 🎙️ Nur Audio transkribieren
```bash
# Einzelne Audio-Datei (MP3, WAV, etc.)
python study_processor_v2.py --input lecture.mp3 --no-screenshots

# Alle Audio-Dateien in einem Ordner
python study_processor_v2.py --input ./audio_files --batch --no-screenshots
```

### 📹 Videos mit Screenshots
```bash
# Einzelnes Video (Standard - empfohlen)
python study_processor_v2.py --input lecture.mp4 --output ./results

# Batch: Alle Videos in einem Ordner
python study_processor_v2.py --input ./videos --batch --output ./results
```

### 📄 Vollständige Analyse mit PDFs
```bash
# Video + Screenshots + PDF-Verknüpfung + HTML-Report
python study_processor_v2.py \
  --input lecture.mp4 \
  --output ./results \
  --studies ./pdf_materials
```

---

## ⚙️ Wichtige Parameter

### Qualität optimieren
```bash
--config configs/lecture_optimized_v2.json    # Beste Erkennungsrate (172+ Wörter)
--model large-v3                              # Bestes Whisper-Modell
--language german                             # Sprache festlegen
```

### Performance anpassen  
```bash
--device cuda                                 # GPU verwenden (schneller)
--cleanup-audio                               # Temporäre Dateien löschen
--batch                                       # Alle Dateien im Ordner
```

### Features ein/ausschalten
```bash
--no-screenshots                              # Screenshots deaktivieren
--no-html                                     # HTML-Report deaktivieren
--similarity-threshold 0.85                   # Screenshot-Sensitivität
```

---

## 📊 Was wird erstellt?

### Ordnerstruktur
```
results/
├── LectureName/
│   ├── LectureName_analysis.json           # 📊 Strukturierte Daten + Timestamps
│   ├── LectureName_report.html            # 🌐 Interaktiver HTML-Report  
│   ├── LectureName_transcript.txt         # 📝 Einfacher Text
│   └── screenshots/                       # 📸 Screenshots mit Zeitstempel
│       ├── LectureName_screenshot_000_00-05-23.jpg
│       └── LectureName_screenshot_001_00-12-45.jpg
└── index.html                             # 📑 Übersichtsseite (bei --batch)
```

### 🌐 HTML-Report Features
- 🔍 **Volltext-Suche** über Transkript und Screenshots
- 📑 **Navigation** zwischen verschiedenen Zeitstellen
- 🖼️ **Screenshot-Timeline** mit präziser Zuordnung  
- 📊 **Qualitätsmetriken** und Statistiken
- 📱 **Mobile-optimiert** für alle Geräte

---

## 🚀 Performance-Tipps

### Modell-Auswahl
| Anwendung | Empfehlung | Grund |
|-----------|------------|-------|
| **Neue Sprecher** | `auto_optimize.py` | 🧠 Automatische Optimierung |
| **Beste Qualität** | `--config lecture_optimized_v2.json` | 🏆 172+ Wörter/Minute |
| **Schnelle Tests** | `--model medium` | ⚡ Guter Kompromiss |
| **Batch-Verarbeitung** | `--config lecture_balanced.json` | ⚖️ Qualität + Geschwindigkeit |

### Effiziente Workflows
```bash
# 1. Optimierung für neuen Professor
python auto_optimize.py --input sample_lecture.mp4 --quick

# 2. Alle weiteren Videos mit optimaler Config  
python study_processor_v2.py --input ./all_lectures --batch --config configs/auto_optimized_*.json

# 3. Große Mengen (RAM sparen)
python study_processor_v2.py --input ./videos --batch --cleanup-audio --device cpu
```

---

## 🔧 Troubleshooting

### Häufige Probleme
```bash
# ❌ Schlechte Transkription → ✅ Auto-Optimierung
python auto_optimize.py --input problematic_video.mp4

# ❌ GPU-Probleme → ✅ CPU verwenden  
python study_processor_v2.py --input video.mp4 --device cpu

# ❌ Speicher-Probleme → ✅ Kleineres Modell
python study_processor_v2.py --input video.mp4 --model medium --cleanup-audio

# ❌ FFmpeg fehlt → ✅ Installation prüfen
ffmpeg -version
```

### Debug & Tests
```bash
# System-Check
python study_processor_v2.py --validate

# Detaillierte Logs
python study_processor_v2.py --input video.mp4 --debug --verbose
```

---

## 📖 Weitere Dokumentation

- **[TRANSCRIPTION_IMPROVEMENTS.md](TRANSCRIPTION_IMPROVEMENTS.md)** - Detaillierte technische Verbesserungen
- **[CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)** - Migration und Bereinigung
- **configs/** - Vordefinierte optimierte Konfigurationen

---

🎉 **Das System lernt automatisch und wird mit jedem Video besser!**