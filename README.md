# 🎓 Study Material Processor v2.0

Ein **intelligentes, selbstlernendes System** zur automatischen Verarbeitung von Vorlesungsvideos mit KI-basierter Transkription, adaptiver Optimierung, intelligenter Screenshot-Extraktion und PDF-Verknüpfung.

## ⚡ Neuste Features

- 🧠 **Adaptive Auto-Optimierung** - Findet automatisch die besten Einstellungen für jeden Sprecher
- 🎯 **Selbstlernende KI** - Wird mit jedem Video besser und lernt verschiedene Sprechstile
- 🔧 **172+ Wörter Erkennungsrate** - Optimiert für deutsche Vorlesungen mit langen Pausen
- 📊 **Audio-Profil Erkennung** - Klassifiziert Sprecher automatisch (dense/moderate/sparse speech)
- 💾 **Optimization Database** - Speichert Lernergebnisse für zukünftige Verwendung

---

## 🚀 Schnellstart

### 🎯 Auto-Optimierung für neue Videos (EMPFOHLEN)
```bash
# Automatische Optimierung für beliebige Sprecher/Module
python auto_optimize.py --input your_lecture.mp4

# Schnelle Optimierung (4 statt 8 Tests)
python auto_optimize.py --input your_lecture.mp4 --quick
```
**→ Erstellt automatisch optimierte Konfiguration für maximale Worterkennnung**

### 📚 Alle 5 Hauptanwendungsfälle

#### 1. 🎙️ Audio-only Transkription
```bash
# Einzelne Audio-Datei
python study_processor_v2.py --input lecture.mp3 --no-screenshots --no-html

# Mehrere Audio-Dateien in Ordner
python study_processor_v2.py --input ./audio_files --batch --no-screenshots
```

#### 2. 📹 Video-only Transkription  
```bash
# Einzelnes Video
python study_processor_v2.py --input lecture.mp4 --no-screenshots

# Mehrere Videos in Ordner
python study_processor_v2.py --input ./videos --batch --no-screenshots
```

#### 3. 📹 + 📸 Video mit Screenshots
```bash
# Screenshots an wichtigen Stellen extrahieren
python study_processor_v2.py --input lecture.mp4 --output ./results
```

#### 4. 📹 + 📸 + 📄 Video + Screenshots + Report (Standard)
```bash
# Vollständiger interaktiver HTML-Report
python study_processor_v2.py --input lecture.mp4 --output ./results --studies ./pdfs
```

#### 5. 📹 + 📸 + 📄 + 🔗 Vollanalyse mit PDF-Verknüpfung
```bash
# Komplette Studienmaterial-Analyse für Frontend
python study_processor_v2.py \
  --input lecture.mp4 \
  --output ./results \
  --studies ./studies \
  --similarity-threshold 0.85 \
  --generate-frontend-data
```

---

## 📋 Inhaltsverzeichnis

- [Installation](#-installation)
- [Auto-Optimierung](#-auto-optimierung-neu)
- [Hauptanwendungsfälle](#-hauptanwendungsfälle)
- [Konfiguration](#️-konfiguration)
- [Ausgabeformate](#-ausgabeformate)
- [Performance & Tipps](#-performance--tipps)
- [Troubleshooting](#-troubleshooting)

---

## 🛠️ Installation

### 1. Python-Abhängigkeiten
```bash
pip install -r requirements.txt
```

### 2. FFmpeg installieren
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download von https://ffmpeg.org/
```

### 3. Setup überprüfen
```bash
python study_processor_v2.py --validate
```

---

## 🧠 Auto-Optimierung (NEU!)

### Automatische Optimierung für jeden Sprecher

Das System analysiert automatisch Audio-Eigenschaften und findet die optimalen Einstellungen:

```bash
# Für neues Modul/Sprecher - automatische Optimierung
python auto_optimize.py --input new_professor_lecture.mp4

# Ergebnis: Optimierte Konfiguration wird erstellt
# Output: configs/auto_optimized_new_professor_lecture_1234567890.json
```

### Verwende optimierte Konfiguration
```bash
# Mit der auto-generierten Konfiguration
python study_processor_v2.py \
  --input weitere_vorlesung.mp4 \
  --config configs/auto_optimized_new_professor_lecture_1234567890.json
```

### Audio-Profil Features
- **Sprecher-Klassifikation**: `dense_speech`, `moderate_speech`, `sparse_speech`, `very_sparse`
- **Pausenlängen-Erkennung**: Automatische Anpassung an Sprechstil
- **Lautstärke-Optimierung**: Adaptive Schwellenwerte
- **Learning Database**: Wird kontinuierlich besser

### Vordefinierte optimierte Profile
```bash
# Basierend auf Testergebnissen (172+ Wörter)
python study_processor_v2.py --input video.mp4 --config configs/lecture_optimized_v2.json  # Beste Qualität
python study_processor_v2.py --input video.mp4 --config configs/lecture_balanced.json      # Ausgewogen 
python study_processor_v2.py --input video.mp4 --config configs/lecture_fast.json         # Schnell
```

---

## 📖 Hauptanwendungsfälle

### 1. 🎙️ Audio-Transkription
**Einzelne oder mehrere Audio-Dateien zu Text**

```bash
# Einzelne Audio-Datei (MP3, WAV, etc.)
python study_processor_v2.py \
  --input lecture.mp3 \
  --output ./results \
  --no-screenshots \
  --no-html

# Alle Audio-Dateien in einem Ordner
python study_processor_v2.py \
  --input ./audio_lectures \
  --batch \
  --no-screenshots \
  --cleanup-audio
```

**Ausgabe**: 
- `results/lecture_transcript.json` - Strukturierte Transkription
- `results/lecture_transcript.txt` - Einfacher Text

### 2. 📹 Video-Transkription
**Video-Dateien zu Text (ohne Screenshots)**

```bash
# Einzelnes Video
python study_processor_v2.py \
  --input lecture.mp4 \
  --output ./results \
  --no-screenshots \
  --model large-v3 \
  --language german

# Mehrere Videos (Batch-Verarbeitung)
python study_processor_v2.py \
  --input ./video_lectures \
  --batch \
  --output ./results \
  --no-screenshots \
  --cleanup-audio
```

**Ausgabe**: 
- JSON mit Transkript + Timestamps
- Optional: HTML-Report

### 3. 📹 + 📸 Video mit Screenshots
**Screenshots an bedeutsamen Stellen extrahieren**

```bash
# Screenshots bei wichtigen Szenen (Tafelbilder, Folien)
python study_processor_v2.py \
  --input lecture.mp4 \
  --output ./results \
  --similarity-threshold 0.85 \
  --min-interval 3.0
```

**Ausgabe**:
- Transkription mit Timestamps
- Screenshots bei Szenenänderungen
- Timeline-Zuordnung Speech ↔ Screenshots

### 4. 📹 + 📸 + 📄 Vollständiger Report (Standard)
**Kompletter interaktiver HTML-Report**

```bash
# Standard-Verarbeitung mit allen Features
python study_processor_v2.py \
  --input lecture.mp4 \
  --output ./results \
  --studies ./pdfs
```

**Ausgabe**:
- 🔍 **Interaktiver HTML-Report** mit Volltext-Suche
- 🖼️ **Screenshot-Timeline** mit Zeitstempel-Zuordnung
- 📊 **Qualitätsmetriken** und Statistiken
- 📱 **Mobile-optimierte Darstellung**

### 5. 📹 + 📸 + 📄 + 🔗 Vollanalyse mit PDF-Verknüpfung
**Komplette Studienmaterial-Analyse für Frontend-Integration**

```bash
# Vollständige Analyse für Studienportal/LMS
python study_processor_v2.py \
  --input lecture.mp4 \
  --output ./results \
  --studies ./studies \
  --similarity-threshold 0.85 \
  --pdf-matching-threshold 0.7 \
  --generate-metadata \
  --export-frontend-json
```

**Ausgabe**:
- 🔗 **PDF-Verknüpfungen** basierend auf Themen/Datum
- 🎯 **Relevanz-Scoring** für beste Treffer  
- 📊 **Frontend-JSON** für Web-Integration
- 🔍 **Keyword-Extraktion** für Suchfunktionen
- 📑 **Content-Vorschau** für PDFs

---

## ⚙️ Konfiguration

### Command-Line Parameter (Häufigste)

#### Eingabe/Ausgabe
```bash
--input VIDEO.mp4          # Eingabedatei oder Ordner
--output ./results         # Ausgabeordner  
--studies ./pdfs          # Ordner mit PDFs/Studienmaterialien
--batch                   # Alle Dateien im Ordner verarbeiten
```

#### Transkriptions-Qualität
```bash
--model large-v3          # Whisper-Modell (tiny|base|small|medium|large|large-v3)
--language german         # Sprache (auto|german|english|...)
--config CONFIG.json      # Benutzerdefinierte Konfiguration
--device cuda             # GPU verwenden (cuda|cpu|auto)
```

#### Features ein/aus
```bash
--no-screenshots         # Screenshots deaktivieren  
--no-html                # HTML-Report deaktivieren
--no-json                # JSON-Export deaktivieren
--cleanup-audio          # Temporäre Audio-Dateien löschen
```

#### Screenshot-Parameter
```bash
--similarity-threshold 0.85    # Schwelle für Szenenänderung (0.0-1.0)
--min-interval 3.0            # Min. Zeit zwischen Screenshots (Sekunden)
```

### Vordefinierte optimierte Konfigurationen

#### Basierend auf Optimierungs-Tests
```bash
# Beste Worterkennnung (172+ Wörter, basierend auf Tests)
python study_processor_v2.py --input video.mp4 --config configs/lecture_optimized_v2.json

# Ausgewogen: Qualität vs. Geschwindigkeit (145 Wörter in 19s)  
python study_processor_v2.py --input video.mp4 --config configs/lecture_balanced.json

# Schnelle Verarbeitung (52 Wörter in 9s)
python study_processor_v2.py --input video.mp4 --config configs/lecture_fast.json

# Enhanced Transcriber (für problematische Audio-Dateien)
python study_processor_v2.py --input video.mp4 --config configs/lecture_fixed.json
```

#### Für spezielle Anforderungen
```bash
# Sehr langsame Sprecher mit langen Pausen
python study_processor_v2.py --input video.mp4 --config configs/slow_speaker.json

# Alte, kompatible Konfiguration  
python study_processor_v2.py --input video.mp4 --config configs/lecture_optimized.json
```

---

## 📊 Ausgabeformate

### Ordnerstruktur
```
results/
├── LectureName/
│   ├── LectureName_analysis.json           # 📊 Strukturierte Daten
│   ├── LectureName_report.html            # 🌐 Interaktiver Report
│   ├── LectureName_transcript.txt         # 📝 Einfacher Text
│   └── screenshots/                       # 📸 Screenshots
│       ├── LectureName_screenshot_000_00-05-23.jpg
│       ├── LectureName_screenshot_001_00-12-45.jpg
│       └── ...
├── AnotherLecture/
│   └── ...
└── index.html                             # 📑 Übersichtsseite (bei --batch)
```

### 🌐 HTML-Report Features
- 🔍 **Volltext-Suche** über Transkript, Screenshots und PDFs
- 📑 **Tab-Navigation** zwischen verschiedenen Inhalten  
- 🖼️ **Screenshot-Timeline** mit präziser Zeitstempel-Zuordnung
- 📄 **PDF-Vorschau** mit automatischer Relevanz-Bewertung
- 📊 **Qualitätsmetriken** und Transkriptions-Statistiken
- 📱 **Mobile-optimiert** für Tablets und Smartphones

### 📊 JSON-Datenformat (für Frontend)
```json
{
  "video_info": {
    "path": "lecture.mp4",
    "duration": 3600.5,
    "title": "Mathematik Vorlesung 12"
  },
  "transcription": {
    "segments": [
      {
        "start": 0.0,
        "end": 5.2, 
        "text": "Guten Morgen zur Vorlesung...",
        "confidence": 0.95
      }
    ],
    "full_text": "Vollständiger Transkript-Text...",
    "word_count": 2847,
    "average_confidence": 0.91
  },
  "screenshots": [
    {
      "timestamp": 123.4,
      "file": "screenshot_001_00-02-03.jpg",
      "related_text": "Hier sehen Sie die Formel..."
    }
  ],
  "related_pdfs": [
    {
      "file": "chapter_12.pdf", 
      "relevance": 0.89,
      "matched_keywords": ["Integral", "Ableitung"],
      "preview": "In diesem Kapitel behandeln wir..."
    }
  ],
  "optimization_metadata": {
    "config_used": "lecture_optimized_v2",
    "speaker_profile": "moderate_speech",
    "processing_time": 127.3
  }
}
```

---

## 🚀 Performance & Tipps

### 🎯 Modell-Auswahl für optimale Ergebnisse

| Szenario | Modell | Begründung |
|----------|--------|-----------|
| **Neue Sprecher/Module** | `auto_optimize.py` | 🧠 Automatische Optimierung |
| **Beste Qualität** | `large-v3` | 🏆 Höchste Genauigkeit |
| **Ausgewogen** | `large` | ⚖️ Guter Kompromiss |
| **Schnelle Tests** | `medium` | ⚡ Moderate Geschwindigkeit |
| **Entwicklung/Debug** | `tiny` | 🔧 Schnellste Verarbeitung |

### 🔧 Optimierte Workflows

#### Neue Sprecher/Module
```bash
# 1. Auto-Optimierung durchführen
python auto_optimize.py --input sample_lecture.mp4 --quick

# 2. Optimierte Config für alle weiteren Videos verwenden
python study_processor_v2.py \
  --input ./all_lectures \
  --batch \
  --config configs/auto_optimized_sample_lecture_*.json
```

#### Batch-Verarbeitung (große Mengen)
```bash
# Effiziente Verarbeitung vieler Videos
python study_processor_v2.py \
  --input ./video_archive \
  --batch \
  --config configs/lecture_balanced.json \
  --cleanup-audio \
  --device cuda \
  --output ./processed
```

#### Schnelle Vorschau
```bash
# Für schnelle Übersicht
python study_processor_v2.py \
  --input lecture.mp4 \
  --config configs/lecture_fast.json \
  --no-screenshots \
  --cleanup-audio
```

### 💾 Speicher-Optimierung
```bash
# Für große Videos oder wenig RAM
python study_processor_v2.py \
  --input huge_lecture.mp4 \
  --model medium \
  --cleanup-audio \
  --device cpu \
  --no-screenshots  # Falls nicht benötigt
```

---

## 🔧 Troubleshooting

### ❗ Häufige Probleme & Lösungen

#### 1. Unvollständige Transkription (Textpassagen fehlen)
```bash
# ✅ Lösung: Auto-Optimierung verwenden
python auto_optimize.py --input problematic_video.mp4

# Oder: Vordefinierte optimierte Konfiguration
python study_processor_v2.py --input video.mp4 --config configs/lecture_optimized_v2.json
```

#### 2. Schlechte Worterkennnung  
```bash
# ✅ Lösung: Größeres Modell + optimierte Konfiguration
python study_processor_v2.py \
  --input video.mp4 \
  --model large-v3 \
  --config configs/lecture_optimized_v2.json
```

#### 3. CUDA/GPU-Probleme
```bash
# ✅ Lösung: CPU erzwingen
python study_processor_v2.py --input video.mp4 --device cpu

# Oder: Spezifische GPU verwenden
python study_processor_v2.py --input video.mp4 --device cuda:0
```

#### 4. Speicher-Probleme (Out of Memory)
```bash
# ✅ Lösung: Kleineres Modell verwenden
python study_processor_v2.py --input video.mp4 --model medium --device cpu
```

#### 5. FFmpeg nicht gefunden
```bash
# ✅ Prüfen ob installiert
ffmpeg -version

# Installation:
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg  
# Windows: Download von https://ffmpeg.org/
```

### 🔍 Debug & Analyse

#### Detaillierte Fehleranalyse
```bash
# Ausführliche Logs für Debugging
python study_processor_v2.py --input video.mp4 --debug --verbose

# Audio-Qualität analysieren
python transcription_analyzer.py --audio video.mp4 --visualize
```

#### Optimierungs-Tests
```bash
# Vergleiche verschiedene Konfigurationen
python optimize_audio_only.py  # Testet 9 verschiedene Einstellungen

# Eigene Optimierung für spezielle Videos
python auto_optimize.py --input special_case.mp4 --max-configs 12
```

### 📊 Qualitätskontrolle
```bash
# System-Check
python study_processor_v2.py --validate

# Performance-Test
python test_video_v2.py  

# Audio-Analyse 
python transcription_analyzer.py --audio your_video.mp4 --compare
```

---

## 🎯 Zusammenfassung

### 🚀 Schnelle Befehle für alle Anwendungsfälle

| Anwendungsfall | Command | Features |
|----------------|---------|----------|
| **🧠 Auto-Optimierung** | `python auto_optimize.py --input video.mp4` | Automatische Optimierung für jeden Sprecher |
| **🎙️ Audio-only** | `python study_processor_v2.py --input audio.mp3 --no-screenshots` | Nur Transkription |
| **📹 Video-only** | `python study_processor_v2.py --input video.mp4 --no-screenshots` | Video → Text |  
| **📸 Mit Screenshots** | `python study_processor_v2.py --input video.mp4` | Text + Screenshots |
| **📄 Vollständiger Report** | `python study_processor_v2.py --input video.mp4 --studies ./pdfs` | Alles + HTML-Report |
| **🔗 Frontend-Integration** | `python study_processor_v2.py --input video.mp4 --studies ./pdfs --export-frontend-json` | Vollanalyse + JSON |
| **⚡ Batch alle Videos** | `python study_processor_v2.py --input ./videos --batch` | Alle Videos in Ordner |
| **🏆 Beste Qualität** | `python study_processor_v2.py --input video.mp4 --config configs/lecture_optimized_v2.json` | 172+ Wörter Erkennnung |

### 🎓 Empfohlener Workflow

1. **Neue Sprecher/Module**: `python auto_optimize.py --input sample.mp4` 
2. **Optimierte Config verwenden**: `python study_processor_v2.py --config auto_optimized_*.json`
3. **Batch-Verarbeitung**: `--batch` für alle Videos
4. **Frontend-Integration**: `--export-frontend-json` für Webanwendungen

---

## 📚 Weitere Dokumentation

- **[TRANSCRIPTION_IMPROVEMENTS.md](TRANSCRIPTION_IMPROVEMENTS.md)** - Detaillierte Verbesserungen der Transkription
- **[CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)** - Migration von v1.0 zu v2.0
- **[src/adaptive_optimizer.py](src/adaptive_optimizer.py)** - Auto-Optimierungs-System
- **[optimization_database.json](optimization_database.json)** - Lern-Datenbank (wird automatisch erstellt)

---

🎉 **Das System wird mit jedem Video intelligenter und optimiert sich automatisch für verschiedene Sprecher und Vorlesungsstile!**