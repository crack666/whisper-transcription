# 📊 Benchmark & Performance Logging

Das Tool verfügt jetzt über ein **zentrales Benchmarking-System**, das automatisch Performance-Daten über alle Runs hinweg sammelt und speichert.

## 🎯 Was wird geloggt?

### Automatisch erfasste Daten

**Pro Run:**
- ⏱️ **Gesamt-Verarbeitungszeit** und Zeit pro Phase
- 📁 **Datei-Informationen**: Größe, Dauer, Auflösung, FPS
- 🔧 **Verwendete Konfiguration**: Modell, Sprache, Device, Segmentierungs-Modus
- 💻 **Hardware-Specs**: CPU, RAM, GPU (falls vorhanden)
- 📊 **Ergebnisse**: Wort-Count, Segment-Count, Screenshot-Count
- 📈 **Performance-Metriken**: RTF, Speedup, Processing Speed, Throughput

**Phasen-Tracking:**
1. `audio_extraction` - Audio aus Video extrahieren
2. `transcription` - Whisper-Transkription
3. `screenshot_extraction` - Screenshots extrahieren
4. `pdf_matching` - PDFs finden
5. `mapping_and_report_generation` - Mappings & HTML-Report

### Berechnete Metriken

```python
{
  "rtf": 0.25,                    # Real-Time Factor (0.25 = 4x faster als Realtime)
  "speedup": 4.0,                 # Wie viel schneller als Realtime
  "processing_speed_x": 4.0,      # Sekunden Media pro Sekunde Processing
  "throughput_mb_per_sec": 12.5,  # MB/s Durchsatz
  "words_per_second": 45.2        # Transkribierte Wörter pro Sekunde
}
```

## 📂 Wo werden Daten gespeichert?

```
benchmark_logs/
└── benchmark_history.jsonl    # JSONL-Format (eine JSON-Zeile pro Run)
```

**JSONL-Format:** Jeder Run ist eine Zeile → einfach zu appenden, einfach zu parsen, niemals korrupt.

## 🔍 Benchmarks anzeigen

### Alle Statistiken anzeigen

```bash
python view_benchmarks.py
```

**Ausgabe:**
```
================================================================================
📊 BENCHMARK REPORT
================================================================================

📈 Overall Statistics:
   Total Runs: 47
   Successful: 45 (95.7%)
   Failed: 2
   Models Used: large-v3, medium, base

⚡ Average Performance:
   rtf: 0.18 (min: 0.12, max: 0.35)
   speedup: 5.67 (min: 2.86, max: 8.33)
   processing_speed_x: 5.67 (min: 2.86, max: 8.33)
   throughput_mb_per_sec: 15.3 (min: 8.2, max: 24.5)

🎯 Performance by Model:
   large-v3:
      Runs: 32
      Success Rate: 96.9%
      Avg Processing Time: 342.5s
      Avg Speedup: 4.82x
   
   medium:
      Runs: 15
      Success Rate: 93.3%
      Avg Processing Time: 218.7s
      Avg Speedup: 7.21x
```

### Nach Modell filtern

```bash
# Nur large-v3 Runs
python view_benchmarks.py --model large-v3

# Nur medium mit CPU
python view_benchmarks.py --model medium --device cpu
```

### Nach Segmentierungs-Modus filtern

```bash
# Nur defensive_silence Mode
python view_benchmarks.py --segmentation-mode defensive_silence

# Nur whole-file (no segmentation)
python view_benchmarks.py --segmentation-mode none
```

### Letzte N Runs anzeigen

```bash
# Letzte 5 Runs
python view_benchmarks.py --last 5
```

### Raw Data Export

```bash
# Als JSON exportieren
python view_benchmarks.py --export stats.json

# Raw data anzeigen
python view_benchmarks.py --raw

# Letzte 3 Runs als JSON
python view_benchmarks.py --raw --last 3
```

## 📊 Beispiel: Performance vergleichen

### Szenario: Welches Modell ist für meine Hardware optimal?

```bash
# 1. Mehrere Runs mit verschiedenen Modellen
python study_processor_v2.py --input test.mp4 --model large-v3
python study_processor_v2.py --input test.mp4 --model medium
python study_processor_v2.py --input test.mp4 --model base

# 2. Statistiken vergleichen
python view_benchmarks.py

# 3. Spezifisches Modell analysieren
python view_benchmarks.py --model medium
```

### Szenario: Segmentation vs. No-Segmentation

```bash
# Test mit und ohne Segmentation
python study_processor_v2.py --input test.mp4
python study_processor_v2.py --input test.mp4 --no-segmentation

# Vergleiche Performance
python view_benchmarks.py --last 2
```

## 🎯 Hardware-spezifische Empfehlungen generieren

Nach mehreren Runs auf Ihrer Hardware können Sie **datenbasierte Empfehlungen** erstellen:

```bash
# Exportiere Statistiken
python view_benchmarks.py --export my_hardware_stats.json

# Analysiere JSON und erstelle Empfehlungen
# Beispiel Python-Script:
```

```python
import json

with open('my_hardware_stats.json') as f:
    stats = json.load(f)

# Finde schnellstes Modell
best_model = max(
    stats['by_model'].items(),
    key=lambda x: x[1].get('avg_speedup', 0)
)

print(f"Empfehlung für Ihre Hardware:")
print(f"  Schnellstes Modell: {best_model[0]}")
print(f"  Durchschnitt Speedup: {best_model[1]['avg_speedup']:.2f}x")
```

## 📈 Integration in README

Die Benchmark-Daten können genutzt werden um **README.md mit echten Zahlen** zu aktualisieren:

**Vorher (geschätzt):**
```markdown
# ⚡ Schnell - 1h Video in ~5-10min
```

**Nachher (gemessen auf Ihrer Hardware):**
```markdown
# ⚡ Schnell - 1h Video in ~6.3min (durchschnittlich auf diesem System)
# Basierend auf 47 Benchmark-Runs mit large-v3 Modell
```

## 🔧 Programmierter Zugriff

```python
from src.benchmark_logger import BenchmarkLogger

# Logger initialisieren
logger = BenchmarkLogger()

# Statistiken abrufen
stats = logger.get_statistics()

# Nach Modell filtern
large_v3_stats = logger.get_statistics(filter_by={'config.model': 'large-v3'})

# Report ausgeben
logger.print_report()

# Spezifische Metrik extrahieren
avg_speedup = stats['average_metrics']['speedup']['mean']
print(f"Durchschnittlicher Speedup: {avg_speedup}x")
```

## 🎨 Daten-Format

### Run-Eintrag (JSONL)

```json
{
  "timestamp": "2025-11-12T15:34:22",
  "file": {
    "path": "/path/to/video.mp4",
    "name": "video.mp4",
    "size_mb": 2450.5,
    "duration_seconds": 3600.0,
    "fps": 30.0,
    "resolution": "1920x1080",
    "type": "video"
  },
  "config": {
    "model": "large-v3",
    "language": "german",
    "device": "cuda",
    "segmentation_enabled": false,
    "segmentation_mode": "none",
    "screenshots_enabled": true
  },
  "hardware": {
    "platform": "Windows-10-10.0.19045",
    "processor": "Intel64 Family 6 Model 142 Stepping 12",
    "cpu_count": 4,
    "cpu_count_logical": 8,
    "ram_gb": 32.0,
    "gpu": {
      "available": true,
      "name": "NVIDIA GeForce RTX 3070",
      "count": 1,
      "cuda_version": "11.8"
    }
  },
  "phases": {
    "audio_extraction": {
      "start": 1699800862.123,
      "end": 1699800865.456,
      "duration_seconds": 3.33
    },
    "transcription": {
      "start": 1699800865.456,
      "end": 1699801245.789,
      "duration_seconds": 380.33,
      "metadata": {
        "segment_count": 267,
        "word_count": 12453
      }
    },
    "screenshot_extraction": {
      "duration_seconds": 332.1,
      "metadata": {
        "screenshot_count": 1636
      }
    }
  },
  "total_duration_seconds": 745.2,
  "success": true,
  "results": {
    "word_count": 12453,
    "segment_count": 267,
    "screenshot_count": 1636,
    "pdf_count": 3
  },
  "metrics": {
    "rtf": 0.207,
    "speedup": 4.83,
    "processing_speed_x": 4.83,
    "throughput_mb_per_sec": 3.29,
    "words_per_second": 16.71
  }
}
```

## 🚀 Automatisierung

### Batch-Testing verschiedener Konfigurationen

```bash
#!/bin/bash
# test_all_configs.sh

TESTFILE="sample.mp4"

# Test alle Modelle
for model in large-v3 medium base; do
    echo "Testing $model..."
    python study_processor_v2.py --input $TESTFILE --model $model
done

# Test mit/ohne Segmentation
python study_processor_v2.py --input $TESTFILE --no-segmentation
python study_processor_v2.py --input $TESTFILE  # mit Segmentation

# Ergebnisse anzeigen
python view_benchmarks.py --last 5
```

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark

on: [push]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmark
        run: |
          python study_processor_v2.py --input test_video.mp4
          python view_benchmarks.py --export benchmark_results.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmark_results.json
```

## 📝 Best Practices

1. **Konsistente Test-Dateien**: Nutze dieselben Test-Dateien für vergleichbare Benchmarks
2. **Mehrere Runs**: Führe mindestens 3-5 Runs pro Konfiguration durch
3. **Hardware-Tags**: Dokumentiere Hardware-Änderungen in separaten Benchmark-Sessions
4. **Regelmäßige Auswertung**: Prüfe Benchmarks nach Updates/Änderungen
5. **Export & Archivierung**: Exportiere wichtige Benchmarks für langfristige Vergleiche

## 🎯 Use Cases

### 1. Hardware-Upgrade Evaluation
Vor/nach RAM-Upgrade → Vergleiche Speedup-Änderungen

### 2. Model Selection
Finde optimales Modell für deine typischen Video-Längen

### 3. Performance Regression Detection
Erkenne Performance-Verschlechterungen nach Code-Änderungen

### 4. Customer Support
"Warum ist es bei mir langsam?" → Vergleiche mit Benchmark-Daten

### 5. Documentation
Ersetze geschätzte Zeiten mit gemessenen Durchschnittswerten

---

**Das Benchmarking-System läuft vollautomatisch im Hintergrund - keine Änderung am Workflow nötig!** 📊✨
