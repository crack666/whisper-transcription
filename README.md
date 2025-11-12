# 🎓 Whisper Transcription Tool v2.2

**Professionelles Transkriptions-Tool** für Vorlesungen, Meetings und Interviews mit KI-Spracherkennung (OpenAI Whisper), automatischer Screenshot-Extraktion und interaktiven HTML-Reports.

## 🚀 Schnellstart

```bash
# Einzelnes Video transkribieren - fertig!
python study_processor_v2.py --input video.mp4

# Ergebnis: Alles im selben Verzeichnis
# → video_report.html (interaktive Timeline)
# → video_transcript.txt (reiner Text)
# → video_analysis.json (strukturierte Daten)
# → video_screenshots/ (automatische Screenshots)
```

**Das war's!** 🎉 Alle Ergebnisse landen automatisch neben Ihrer Quelldatei.

## ⚡ Was macht dieses Tool?

**Sie geben:** `video.mp4` oder `audio.mp3`

**Sie bekommen automatisch:**
- 📝 **Vollständiges Transkript** mit Zeitstempeln
- 🖼️ **Intelligente Screenshots** bei Folienwechseln
- 🌐 **Interaktiver HTML-Report** mit Timeline und Suche
- 📄 **Plain-Text-Export** für Copy-Paste
- 📊 **JSON-Daten** für programmatische Nutzung

### Hauptmerkmale

- ✅ **Plug & Play** - Einfach starten, keine Konfiguration
- ⚡ **Schnell** - 1h Video in ~5-10min (mit `--no-segmentation`)
- 🎯 **Präzise** - OpenAI Whisper `large-v3`
- 📂 **Organisiert** - Alle Dateien sauber strukturiert
- 🔄 **Batch-fähig** - Ganze Ordner auf einmal

## � Einfache Anwendungsbeispiele

### Basis-Nutzung (ein Kommando!)

```bash
# Einzelnes Video transkribieren
python study_processor_v2.py --input lecture.mp4

# Einzelne Audio-Datei
python study_processor_v2.py --input interview.mp3

# Ganzen Ordner verarbeiten
python study_processor_v2.py --input ./videos/ --batch
```

### Häufige Use Cases

```bash
# � Maximale Geschwindigkeit (moderne Hardware)
python study_processor_v2.py --input video.mp4 --no-segmentation

# 📂 Mit eigenem Output-Ordner
python study_processor_v2.py --input video.mp4 --output ./ergebnisse

# 🎯 Nur Audio, keine Screenshots
python study_processor_v2.py --input audio.mp3 --no-screenshots

# 📚 Batch mit allen Optionen
python study_processor_v2.py --input ./vorlesungen/ --batch --no-segmentation --output ./results
```

### Was Sie bekommen

```
📁 Ihr Video-Verzeichnis/
├── video.mp4                        # Ihr Original
├── video_transcript.txt             # ✨ Reiner Text mit Zeitstempeln
├── video_report.html                # ✨ Interaktive Timeline + Suche
├── video_analysis.json              # ✨ Alle Daten strukturiert
└── 📁 video_screenshots/            # ✨ Automatische Screenshots
    ├── screenshot_000_00-00-05.jpg
    ├── screenshot_001_00-02-34.jpg
    └── ...
```

**Komplett automatisch. Keine Konfiguration nötig.** 🎉

---

## ⚡ Performance-Optionen

### Geschwindigkeit vs. Kompatibilität

```bash
# 🐢 Standard-Modus (sicher, kompatibel)
python study_processor_v2.py --input video.mp4
# → ~15-20 min für 1h Video

# 🚀 Performance-Modus (modern, schnell)
python study_processor_v2.py --input video.mp4 --no-segmentation
# → ~5-10 min für 1h Video (3-7x schneller!)
```

**Empfehlung:**
- **Standard-Modus:** Ältere Hardware, sehr lange Videos (>2h), maximale Stabilität
- **Performance-Modus:** Moderne Hardware (≥16GB RAM), Batch-Processing, Produktions-Workflows

### Modell-Auswahl

```bash
# 🏆 Beste Qualität (Standard)
python study_processor_v2.py --input video.mp4 --model large-v3

# ⚡ Schneller, gute Qualität
python study_processor_v2.py --input video.mp4 --model medium

# 🚀 Maximaler Speed
python study_processor_v2.py --input video.mp4 --model medium --no-segmentation
```

| Modell | Qualität | Geschwindigkeit | Empfohlen für |
|--------|----------|-----------------|---------------|
| `large-v3` | ⭐⭐⭐⭐⭐ | Normal | Beste Ergebnisse |
| `large-v3-turbo` | ⭐⭐⭐⭐⭐ | Schneller | Neu! Schneller als large-v3 |
| `medium` | ⭐⭐⭐⭐ | Schneller | Guter Kompromiss |
| `base` | ⭐⭐⭐ | Schnellste | Tests |

**⚠️ Hinweis:** `large-v3-turbo` ist ein neues Modell (Nov 2024). Nutze Benchmarking um die Performance auf deiner Hardware zu messen!

---

## �️ Installation

```bash
# 1. Repository klonen
git clone <repository-url>
cd whisper-transcription

# 2. Dependencies installieren
pip install -r requirements.txt

# 3. FFmpeg installieren (falls noch nicht vorhanden)
# Windows: https://ffmpeg.org/download.html
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# 4. Test
python study_processor_v2.py --validate
```

**Fertig!** Sie können jetzt Videos transkribieren. 🚀

## 🎯 Erweiterte Anwendungsfälle

### Spezielle Szenarien

```bash
# 📄 Mit PDF-Verknüpfung (findet relevante Dokumente)
python study_processor_v2.py --input lecture.mp4 --studies ./pdf_materials

# 🌍 Andere Sprache
python study_processor_v2.py --input video.mp4 --language english

# �️ GPU-Beschleunigung nutzen
python study_processor_v2.py --input video.mp4 --device cuda

# 🧹 Temporäre Dateien aufräumen
python study_processor_v2.py --input video.mp4 --cleanup-audio

# 🎨 Screenshot-Sensitivität anpassen
python study_processor_v2.py --input video.mp4 --similarity-threshold 0.90
```

### Utility-Tools (optional)

```bash
# 🔄 Screenshots nachträglich regenerieren
python regenerate_screenshots.py "video_analysis.json"

# 📄 HTML-Report neu erstellen
python regenerate_report.py

# 📝 Text separat extrahieren (bereits automatisch, aber für Legacy-Workflows)
python extract_transcript_text.py --input video_analysis.json --timestamps
```

---

## 📊 Output-Beispiele

### 📝 Text-Transkript (`video_transcript.txt`)
```
================================================================================
TRANSCRIPTION
================================================================================

Language: de
Duration: 104.8 minutes (6288.0 seconds)
Segments: 267
Words: 12453

================================================================================

[00:00:00] Guten Morgen zusammen, herzlich willkommen zur heutigen Vorlesung...
[00:05:23] Wie Sie auf der Folie sehen können, haben wir drei Hauptpunkte...
[00:12:45] Das ist ein sehr wichtiger Aspekt, den wir uns genauer ansehen...
```

### 🌐 HTML-Report Features

Der interaktive HTML-Report bietet:
- **📍 Timeline-Navigation** - Durch alle Segmente scrollen
- **🔍 Volltext-Suche** - Schnell bestimmte Stellen finden
- **🖼️ Screenshot-Sync** - Automatische Anzeige passender Screenshots
- **📊 Statistiken** - Dauer, Wörter, Confidence-Werte
- **📱 Responsive** - Funktioniert auf allen Geräten

### 📂 Ordnerstruktur

**Standard (ohne --output):**
```
📁 videos/
├── lecture.mp4                    # Original
├── lecture_transcript.txt         # Text mit Zeitstempeln
├── lecture_report.html            # Interaktiver Report
├── lecture_analysis.json          # Strukturierte Daten
└── 📁 lecture_screenshots/        # Screenshots getrennt
    ├── screenshot_000.jpg
    └── ...
```

**Mit --output:**
```
📁 results/
└── 📁 lecture/
    ├── lecture_transcript.txt
    ├── lecture_report.html
    ├── lecture_analysis.json
    └── 📁 screenshots/
        └── ...
```

---

## ⚙️ Alle Parameter (Referenz)

### Häufig verwendet
```bash
--input FILE/DIR           # Video/Audio-Datei oder Ordner (ERFORDERLICH)
--output DIR               # Ausgabe-Verzeichnis (Standard: wie Input)
--batch                    # Alle Dateien im Input-Ordner verarbeiten
--no-segmentation          # Performance-Modus (3-7x schneller)
--model NAME               # large-v3 (Standard), medium, base
--language LANG            # german (Standard), english, etc.
```

### Weitere Optionen
```bash
--device TYPE              # cuda, cpu (Standard: auto)
--no-screenshots           # Screenshots deaktivieren
--no-html                  # HTML-Report deaktivieren
--similarity-threshold N   # Screenshot-Sensitivität (0.0-1.0, Standard: 0.85)
--min-interval N           # Min. Sekunden zwischen Screenshots (Standard: 2.0)
--cleanup-audio            # Temporäre Audio-Dateien löschen
--studies DIR              # PDF-Verzeichnis für Dokumenten-Matching
--config FILE              # Eigene Konfigurationsdatei
--verbose                  # Detaillierte Logs
--debug                    # Debug-Modus
--validate                 # System-Check ohne Verarbeitung
```

---

## � Tipps & Tricks

### Optimale Einstellungen finden

```bash
# Für neuen Sprecher/Content automatisch optimieren
python auto_optimize.py --input sample.mp4 --quick

# Dann für alle weiteren Videos nutzen
python study_processor_v2.py --input weitere/ --batch \
  --config configs/auto_optimized_*.json
```

### Performance vs. Qualität

| Szenario | Kommando | Zeit (1h Video) |
|----------|----------|-----------------|
| 🐢 **Maximum Kompatibilität** | Standard | ~15-20 min |
| ⚡ **Balanced** | `--model medium` | ~10-15 min |
| 🚀 **Maximum Speed** | `--model medium --no-segmentation` | ~3-5 min |
| 🏆 **Maximum Qualität** | `--model large-v3` | ~15-25 min |
| 🔥 **Optimal** | `--model large-v3 --no-segmentation` | ~5-10 min |

### Batch-Verarbeitung große Mengen

```bash
## 💡 Tipps & Tricks

### Batch-Verarbeitung

```bash
# Empfohlener Workflow für viele Videos
python study_processor_v2.py \
  --input ./semester_videos/ \
  --batch \
  --no-segmentation \
  --cleanup-audio \
  --device cuda
```

### 📊 Performance Benchmarking

Das Tool sammelt **automatisch Performance-Daten** für alle Runs:

```bash
# Statistiken anzeigen
python view_benchmarks.py

# Nach Modell filtern
python view_benchmarks.py --model large-v3

# Letzte 5 Runs
python view_benchmarks.py --last 5

# Als JSON exportieren
python view_benchmarks.py --export stats.json
```

**Nutzen:**
- 🎯 Finde das **optimale Modell für deine Hardware**
- 📈 Vergleiche **echte Performance-Daten** statt Schätzungen
- 💻 Erstelle **hardware-spezifische Empfehlungen**
- 📊 Erkenne **Performance-Regressionen** nach Updates

Siehe [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md) für Details.

### Performance vs. Qualität

**⚠️ Hinweis:** Die folgenden Zeiten sind Richtwerte. Nutze `view_benchmarks.py` für **deine echten Hardware-Daten**.

| Szenario | Kommando | Zeit (1h Video) |
|----------|----------|-----------------|
| Maximum Kompatibilität | Standard | ~15-20 min |
| Balanced | `--model medium` | ~10-15 min |
| Maximum Speed | `--model medium --no-segmentation` | ~3-5 min |
| Maximum Qualität | `--model large-v3` | ~15-25 min |
| Optimal | `--model large-v3 --no-segmentation` | ~5-10 min |
```

---

## 📚 Häufige Fragen (FAQ)

### Welches Modell soll ich verwenden?
- **Empfohlen:** `large-v3` (Standard) - beste Qualität
- **Schneller:** `medium` - guter Kompromiss
- **Tests:** `base` - am schnellsten

### Brauche ich eine GPU?
- **Nein** - funktioniert auch mit CPU (langsamer)
- **Ja, hilft** - mit CUDA-GPU 2-3x schneller
- **Auto-Erkennung** - Tool wählt automatisch beste Option

### Wo landen meine Dateien?
- **Ohne --output:** Direkt neben der Quelldatei ✅ (empfohlen)
- **Mit --output:** In angegebenem Verzeichnis

### Wie groß sollte mein RAM sein?
- **Standard-Modus:** 8GB ausreichend
- **Performance-Modus:** 16GB empfohlen
- **Lange Videos (>2h):** 16-32GB

### Unterstützte Formate?
- **Video:** MP4, AVI, MKV, MOV, WEBM, FLV, WMV
- **Audio:** MP3, WAV, M4A, FLAC, OGG

### Wie lange dauert die Verarbeitung?
- **Standard:** ~15-20 min für 1h Video (geschätzt)
- **Performance-Modus:** ~5-10 min für 1h Video (geschätzt)
- **📊 Für echte Werte:** `python view_benchmarks.py` nach einigen Runs

### Was ist Benchmarking?
- **Automatisch:** Jeder Run wird geloggt (Zeit, Hardware, Modell)
- **Analyse:** `python view_benchmarks.py` zeigt Performance-Stats
- **Nutzen:** Finde optimale Einstellungen für **deine** Hardware
- **Details:** Siehe [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md)
- **Abhängig von:** Hardware, Modell, Segmentierung

---