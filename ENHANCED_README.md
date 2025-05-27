# Enhanced Study Material Processor

Ein erweitertes System zur automatischen Verarbeitung von Vorlesungsvideos mit Transkription, Screenshot-Extraktion und PDF-Verknüpfung.

## Neue Funktionen

### 🎥 Video-Analyse
- Automatische Extraktion von Screenshots bei Tafelbildänderungen
- Intelligente Szenenerkennnung basierend auf Bildähnlichkeit
- Konfigurierbare Schwellenwerte für optimale Ergebnisse

### 📝 Erweiterte Transkription
- Vollständig integriert mit dem bestehenden Whisper-System
- Zeitstempel-basierte Zuordnung von Screenshots zu Transkript-Segmenten
- Unterstützung für alle Whisper-Modelle

### 📚 PDF-Integration
- Automatische Erkennung verwandter PDF-Dateien
- Datums- und Namensbasierte Zuordnung
- Content-Vorschau aus PDF-Dateien

### 🔍 Durchsuchbare HTML-Berichte
- Interaktive HTML-Berichte mit Suchfunktion
- Tabbed Interface für bessere Navigation
- Screenshot-Transkript-Zuordnung in übersichtlicher Darstellung

## Installation

1. Bestehende Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

2. FFmpeg sicherstellen (für Video-Verarbeitung):
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download von https://ffmpeg.org/download.html
```

## Verwendung

### Einfache Verarbeitung aller Study-Videos
```bash
python process_studies.py
```

### Einzelnes Video verarbeiten
```bash
python study_material_processor.py --input "studies/Aufzeichnung - 01.04.2025.mp4" --output_dir "./output" --studies_dir "./studies"
```

### Alle Videos im studies-Ordner verarbeiten
```bash
python study_material_processor.py --input "./studies" --batch_process --output_dir "./output" --studies_dir "./studies" --extract_screenshots --verbose
```

### Erweiterte Parameter
```bash
python study_material_processor.py \
  --input "./studies" \
  --batch_process \
  --output_dir "./output" \
  --studies_dir "./studies" \
  --extract_screenshots \
  --similarity_threshold 0.80 \
  --min_time_between_shots 3.0 \
  --model large-v3 \
  --language german \
  --cleanup_audio \
  --verbose
```

## Parameter

| Parameter | Beschreibung | Standard | Optionen |
|-----------|-------------|----------|----------|
| `--input` | Video-Datei oder Verzeichnis | Required | Dateipfad |
| `--output_dir` | Ausgabeverzeichnis | "./output" | Verzeichnispfad |
| `--studies_dir` | Verzeichnis mit Studienmaterialien | "./studies" | Verzeichnispfad |
| `--extract_screenshots` | Screenshots extrahieren | False | Flag |
| `--similarity_threshold` | Schwellenwert für Szenenänderung | 0.85 | 0.0-1.0 |
| `--min_time_between_shots` | Min. Zeit zwischen Screenshots (s) | 2.0 | Sekunden |
| `--language` | Sprache für Transkription | "german" | Siehe LANGUAGE_MAP |
| `--model` | Whisper-Modell | "large-v3" | tiny, base, small, medium, large, large-v2, large-v3 |
| `--device` | Gerät für Verarbeitung | auto | cpu, cuda, cuda:0, etc. |
| `--batch_process` | Alle Videos im Verzeichnis | False | Flag |
| `--cleanup_audio` | Audio-Dateien nach Verarbeitung löschen | False | Flag |
| `--verbose` | Ausführliche Ausgabe | False | Flag |

## Ausgabe-Struktur

Für jedes verarbeitete Video wird ein Verzeichnis erstellt:

```
output/
├── Aufzeichnung - 01.04.2025/
│   ├── Aufzeichnung - 01.04.2025_analysis.json    # Vollständige Analysedaten
│   ├── Aufzeichnung - 01.04.2025_report.html      # Durchsuchbarer HTML-Bericht
│   └── screenshots/                                # Extrahierte Screenshots
│       ├── Aufzeichnung - 01.04.2025_screenshot_000_00-05-23.jpg
│       ├── Aufzeichnung - 01.04.2025_screenshot_001_00-12-45.jpg
│       └── ...
├── Aufzeichnung - 08.04.2025/
│   └── ...
└── index.html                                      # Übersichtsseite aller Videos
```

## HTML-Berichte

Die generierten HTML-Berichte bieten:

- **Suchfunktion**: Volltext-Suche über Transkript, Screenshots und PDFs
- **Tab-Navigation**: Getrennte Ansichten für verschiedene Inhaltstypen
- **Interaktive Zuordnung**: Screenshots mit entsprechenden Transkript-Segmenten
- **PDF-Integration**: Vorschau und Relevanz-Bewertung verwandter Dokumente

## Beispiel-Workflow

1. **Videos in studies-Ordner ablegen**:
   ```
   studies/
   ├── Aufzeichnung - 01.04.2025.mp4
   ├── Aufzeichnung - 08.04.2025.mp4
   ├── 1_fakultäten_und_binomialkoeffizenten(2).pdf
   └── 2_protokoll.20250401-1aa.pdf
   ```

2. **Verarbeitung starten**:
   ```bash
   python process_studies.py
   ```

3. **Ergebnisse ansehen**:
   - Öffne `output/index.html` für Übersicht
   - Klicke auf Video-Links für detaillierte Berichte
   - Nutze Suchfunktion für spezifische Inhalte

## Funktionsweise

### Screenshot-Extraktion
1. Video wird Sekunde für Sekunde analysiert
2. Bildähnlichkeit zwischen aufeinanderfolgenden Frames berechnet
3. Bei Unterschreitung des Schwellenwerts wird Screenshot erstellt
4. Mindestabstand zwischen Screenshots wird eingehalten

### PDF-Zuordnung
1. Datums-Matching: Extraktion von Daten aus Dateinamen
2. Keyword-Matching: Vergleich gemeinsamer Begriffe
3. Relevanz-Scoring: Bewertung der Ähnlichkeit
4. Content-Extraktion: Vorschau der ersten Seiten

### Transkript-Screenshot-Zuordnung
1. Zeitstempel der Screenshots werden mit Transkript-Segmenten verglichen
2. Nächstliegender Transkript-Abschnitt wird zugeordnet
3. Zeitdifferenz wird dokumentiert
4. Visuelle Darstellung in HTML-Bericht

## Leistungsoptimierung

- **GPU-Nutzung**: Automatische CUDA-Erkennung für Whisper
- **Parallel-Verarbeitung**: Mehrere Audio-Segmente gleichzeitig
- **Memory-Management**: Frames werden nur temporär im Speicher gehalten
- **Caching**: Bereits extrahierte Audio-Dateien werden wiederverwendet

## Troubleshooting

### Video-Codec-Probleme
```bash
# Konvertierung für kompatible Formate
ffmpeg -i input_video.mp4 -c:v libx264 -c:a aac output_video.mp4
```

### Speicher-Probleme bei großen Videos
- Kleineres Whisper-Modell verwenden (`--model medium`)
- Längere Abstände zwischen Screenshots (`--min_time_between_shots 5.0`)
- Höheren Ähnlichkeits-Schwellenwert (`--similarity_threshold 0.90`)

### PDF-Verarbeitung-Fehler
- Stellen Sie sicher, dass PDFs nicht passwortgeschützt sind
- Bei Encoding-Problemen: PDFs neu erstellen oder konvertieren

## Bekannte Limitierungen

- Video-Codecs müssen von OpenCV unterstützt werden
- PDF-Text-Extraktion funktioniert nur bei text-basierten PDFs
- Sehr große Videos (>2GB) können viel Speicher benötigen
- Screenshot-Qualität hängt von Video-Auflösung ab

## Integration mit bestehendem System

Das neue System ist vollständig kompatibel mit dem ursprünglichen `audio_transcription.py`. Alle bestehenden Parameter und Funktionen bleiben verfügbar.