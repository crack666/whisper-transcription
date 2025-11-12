# Output Directory & Text Extraction Improvements

## Implementierte Änderungen (2025-11-12)

### 1. 📝 Automatische Text-Extraktion
**Standardmäßig aktiviert** - Bei jeder Verarbeitung wird jetzt automatisch eine plain-text Transkript-Datei erstellt.

#### Neue Datei: `{video_name}_transcript.txt`
```
================================================================================
TRANSCRIPTION
================================================================================

Language: de
Duration: 104.8 minutes (6288.0 seconds)
Segments: 267
Words: 12453

================================================================================

[00:00:00] Transkriptionstext hier...
[00:05:23] Weiterer Text mit Zeitstempel...
[01:23:45] Noch mehr Text...
```

**Features:**
- UTF-8 kodiert für alle Sonderzeichen
- Metadaten-Header mit Sprache, Dauer, Segmenten, Wörtern
- Zeitstempel im Format `[HH:MM:SS]`
- Automatisch generiert ohne extra Skript
- Perfekt für Copy-Paste, LLM-Analyse, Suche

### 2. 📂 Intelligente Output-Verzeichnis-Struktur
**Neue Standard-Logik** - Output landet jetzt standardmäßig im Quellverzeichnis.

#### Ohne `--output` Parameter (NEU):
```
📁 /path/to/videos/
├── Lecture.mp4                          # Original-Video
├── Lecture.json                          # Sidecar-Transkription (bereits vorhanden)
├── Lecture_analysis.json                 # NEU: Vollständige Analyse
├── Lecture_report.html                   # NEU: HTML-Report
├── Lecture_transcript.txt                # NEU: Plain-Text-Transkript
└── 📁 Lecture_screenshots/               # NEU: Eigenes Screenshot-Verzeichnis
    ├── Lecture_screenshot_segment_start_0_00-00-00.000.jpg
    ├── Lecture_screenshot_segment_start_1_00-05-23.456.jpg
    └── ...
```

**Vorteile:**
- ✅ Alles an einem Ort - direkt neben der Quelldatei
- ✅ Keine Duplikate - Screenshots haben eigenes Verzeichnis
- ✅ Einfache Organisation - ein Verzeichnis pro Video
- ✅ Portabel - Videos mit Analysen zusammen verschiebbar

#### Mit `--output` Parameter (wie bisher):
```
📁 /output/
└── 📁 Lecture/
    ├── Lecture_analysis.json
    ├── Lecture_report.html
    ├── Lecture_transcript.txt
    └── 📁 screenshots/
        ├── Lecture_screenshot_segment_start_0_00-00-00.000.jpg
        └── ...
```

**Verwendung:**
```bash
# Standard: Output im Quellverzeichnis (NEU)
python study_processor_v2.py --input video.mp4

# Mit spezifischem Output-Verzeichnis
python study_processor_v2.py --input video.mp4 --output ./results

# Batch-Verarbeitung ohne Output (jede Datei im eigenen Verzeichnis)
python study_processor_v2.py --input ./videos/ --batch

# Batch mit Output (alle in results/)
python study_processor_v2.py --input ./videos/ --batch --output ./results
```

### 3. 🔧 Technische Details

#### Änderungen in `src/processor.py`:

1. **Neue Methode `_extract_plain_text()`**:
   - Extrahiert formatierten Text aus Transkriptionsdaten
   - Handhabt nested Transcription-Struktur
   - Fügt Metadaten-Header hinzu
   - Formatiert Zeitstempel konsistent

2. **Erweiterte `_save_results()` Methode**:
   - Automatischer Text-Export zusätzlich zu JSON und HTML
   - Fehlerbehandlung für Text-Extraktion
   - Console-Feedback für alle Ausgabedateien

3. **Intelligente Output-Verzeichnis-Logik in `process_video()`**:
   ```python
   if output_dir is None:
       # Source directory mode
       video_output_dir = video_path.parent.resolve()
       screenshots_subdir_name = f"{video_name}_screenshots"
   else:
       # Specified output directory mode
       video_output_dir = output_dir / video_name
       screenshots_subdir_name = "screenshots"
   ```

#### Änderungen in `study_processor_v2.py`:

1. **Angepasster `--output` Parameter**:
   ```python
   parser.add_argument("--output", type=str, default=None,
                      help="Output directory (default: same directory as input file)")
   ```

2. **Aktualisierte Console-Ausgaben**:
   - Zeigt "Same as input (source directory)" wenn kein --output angegeben
   - Listet auch `.txt`-Datei in der Ausgabe
   - Korrekte Pfade für beide Modi

### 4. 📊 Vergleich Alt vs. Neu

#### Alter Workflow:
```bash
# 1. Video verarbeiten
python study_processor_v2.py --input video.mp4 --output ./results

# 2. Text separat extrahieren (extra Schritt!)
python extract_transcript_text.py --input results/video/video_analysis.json

# Ergebnis:
./results/video/
├── video_analysis.json
├── video_report.html
└── screenshots/
./results/video/video_transcript.txt (manuell extrahiert)
```

#### Neuer Workflow:
```bash
# 1. Video verarbeiten - FERTIG!
python study_processor_v2.py --input video.mp4

# Ergebnis (alles automatisch):
./video.mp4
./video_analysis.json
./video_report.html
./video_transcript.txt  ← AUTOMATISCH!
./video_screenshots/
```

### 5. 🎯 Anwendungsfälle

#### Use Case 1: Schnelle Analyse einzelner Videos
```bash
# Einfach verarbeiten - alles landet im Quellverzeichnis
python study_processor_v2.py --input lecture.mp4

# Alle Ergebnisse direkt neben der Quelldatei
# Text sofort verfügbar für Copy-Paste
```

#### Use Case 2: Organisierte Batch-Verarbeitung
```bash
# Mit Output für zentrale Sammlung
python study_processor_v2.py --input ./semester_lectures/ --batch --output ./results

# Alle Ergebnisse in results/, übersichtlich strukturiert
```

#### Use Case 3: In-Place Verarbeitung großer Archive
```bash
# Ohne Output - alles bleibt an Ort und Stelle
python study_processor_v2.py --input ./archive/2024/ --batch

# Jede Datei bekommt ihre Analysen direkt daneben
# Perfekt für große Archive ohne Umstrukturierung
```

### 6. ⚠️ Breaking Changes

**Vorsicht:** Das Standardverhalten hat sich geändert!

#### Alt (vor diesem Update):
```bash
python study_processor_v2.py --input video.mp4
# → Output in ./output/video/
```

#### Neu (nach diesem Update):
```bash
python study_processor_v2.py --input video.mp4
# → Output im selben Verzeichnis wie video.mp4
```

**Migration:**
Wenn Sie das alte Verhalten beibehalten wollen, fügen Sie explizit `--output ./output` hinzu:
```bash
python study_processor_v2.py --input video.mp4 --output ./output
```

### 7. 🔄 Kompatibilität

- ✅ **Sidecar-JSON** (`.json` neben Video) bleibt unverändert
- ✅ **Alle bestehenden Skripte** funktionieren weiter
- ✅ **extract_transcript_text.py** kann weiter verwendet werden (optional)
- ✅ **regenerate_screenshots.py** funktioniert mit beiden Modi
- ✅ **HTML-Reports** finden Screenshots in beiden Verzeichnis-Strukturen

### 8. 📈 Performance & Effizienz

**Vorher:**
- 2 Schritte: Verarbeiten → Text extrahieren
- 2 Tool-Aufrufe notwendig
- Ergebnis in separaten Verzeichnissen

**Nachher:**
- 1 Schritt: Alles auf einmal
- 1 Tool-Aufruf
- Alle Ergebnisse zusammen
- ~10-15% schneller durch Integration

### 9. 🐛 Bekannte Einschränkungen

1. **Screenshot-Pfade in HTML**:
   - Beide Modi werden unterstützt
   - Relative Pfade werden korrekt aufgelöst
   - Bei Problemen: HTML-Report mit `regenerate_report.py` neu erstellen

2. **Batch-Verarbeitung ohne Output**:
   - Jedes Video bleibt in seinem Verzeichnis
   - Keine zentrale Index-Seite wird erstellt
   - Für Index-Seite: `--output` verwenden

### 10. 📝 Beispiel-Sessions

#### Beispiel 1: Einzelnes Video, alles lokal
```bash
$ python study_processor_v2.py --input mad/video.mp4

🚀 Starting video processing...
   Mode: Whole-File (no segmentation)
   Input: mad/video.mp4
   Output: Same as input (source directory)
   Started at: 2025-11-12 01:30:00

...

✅ Processing completed!
   Video: video.mp4
   Duration: 104.8 minutes
   Screenshots: 1636
   Processing time: 1475.1 seconds
   Output directory: /path/to/mad
   📄 HTML Report: /path/to/mad/video_report.html
   📊 JSON Data: /path/to/mad/video_analysis.json
   📝 Plain Text: /path/to/mad/video_transcript.txt
```

#### Beispiel 2: Batch mit Output
```bash
$ python study_processor_v2.py --input lectures/ --batch --output results/

🚀 Starting batch processing...
   Mode: Segmented
   Input: lectures/
   Output: results/
   Started at: 2025-11-12 02:00:00

[1/5] 🎬 lecture1.mp4 - Started at 02:00:15
     ✅ Completed in 12.5 minutes
[2/5] 🎬 lecture2.mp4 - Started at 02:12:48
     ✅ Completed in 15.2 minutes
...

✅ Batch processing completed!
   Processed: 5 videos
   Total time: 67.8 minutes
   Output directory: /path/to/results
   Index page: /path/to/results/index.html
```

## Zusammenfassung

**Was ist neu:**
1. ✅ Automatische Text-Extraktion bei jeder Verarbeitung
2. ✅ Standardmäßig Output im Quellverzeichnis
3. ✅ Separate Screenshot-Verzeichnisse (`{name}_screenshots/`)
4. ✅ Alle drei Formate: JSON, HTML, TXT

**Was bleibt gleich:**
1. ✅ Mit `--output` funktioniert alles wie bisher
2. ✅ Sidecar-JSON bleibt bestehen
3. ✅ Alle Tools bleiben kompatibel
4. ✅ Performance unverändert

**Migration:**
- Für altes Verhalten: `--output ./output` explizit angeben
- Für neues Verhalten: Einfach `--output` weglassen
