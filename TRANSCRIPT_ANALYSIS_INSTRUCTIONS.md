# Handlungsanweisung: Analyse von Vorlesungstranskripten

## Ziel
Erstelle strukturierte Zusammenfassungen aller Vorlesungstranskripte in den Ordnern `mad/`, `pqm/` und `datenbanken/`. Jeder Ordner erhält eine eigene Zusammenfassungsdatei mit den wichtigsten Inhalten aller darin enthaltenen Vorlesungen.

## Dateien und Struktur

### Input-Dateien
- **Verzeichnisse**: `mad/`, `pqm/`, `datenbanken/`
- **Dateiformat**: `*.mp4.txt` (z.B. `Webkonferenz Bbb - 9.10.2025.mp4.txt`)
- **Inhalt**: Vollständige Transkripte der Vorlesungen

### Output-Dateien
Erstelle für jeden Ordner eine Zusammenfassungsdatei:
- `mad/MAD_Zusammenfassung.md`
- `pqm/PQM_Zusammenfassung.md`
- `datenbanken/Datenbanken_Zusammenfassung.md`

## Analysevorgehen (Schritt-für-Schritt)

### Schritt 1: Dateien identifizieren
```
Führe aus: Suche alle .mp4.txt Dateien im Zielverzeichnis
Sortiere nach Datum (extrahiere aus Dateinamen)
Liste die gefundenen Dateien auf
```

### Schritt 2: Einzelne Transkripte analysieren
Für jedes Transkript extrahiere:

#### A. Metadaten
- Datum (aus Dateinamen: z.B. "9.10.2025")
- Titel/Thema der Vorlesung (falls erkennbar)

#### B. Kernthemen (Main Topics)
Identifiziere die 3-5 Hauptthemen, die in der Vorlesung behandelt wurden:
- Technische Konzepte (z.B. "B-Bäume", "Hash-Joins", "Indizes")
- Theoretische Grundlagen
- Praktische Beispiele
- Diskussionen

#### C. Wichtige Inhalte (Key Takeaways)
Extrahiere stichpunktartig:
- Definitionen wichtiger Begriffe
- Erklärte Konzepte und deren Zusammenhänge
- Demonstrierte Beispiele
- Best Practices
- Häufige Fehler / Fallstricke

#### D. **KRITISCH: Aufgaben und Termine**
⚠️ **HÖCHSTE PRIORITÄT** - Suche explizit nach:
- **Hausaufgaben / Einsendeaufgaben**
  - Aufgabennummer (z.B. "Aufgabe 2")
  - Beschreibung der Aufgabe
  - Anforderungen / Erwartungen
  - Bewertungskriterien
  
- **Termine und Fristen**
  - Abgabetermine (z.B. "bis zum 15.11.")
  - Klausurtermine
  - Präsentationstermine
  - Sprechstundentermine
  
- **Hinweise zur Bearbeitung**
  - Welche Daten/Tools verwendet werden sollen
  - Tipps vom Dozenten
  - Häufige Fragen und Antworten
  
- **Bewertungsverfahren**
  - Wie werden Aufgaben bewertet (z.B. "zweistufig: ok / teilweise ok / nicht ok")
  - Möglichkeit zur Nachbesserung
  - Bestehensvoraussetzungen

#### E. Diskussionen und Fragen
- Studentenfragen und Dozentenantworten
- Klärungen von Unklarheiten
- Zusätzliche Erläuterungen

#### F. Ausblick
- Angekündigte Themen für nächste Sitzung
- Verweise auf Skriptkapitel
- Empfohlene Literatur

### Schritt 3: Zusammenfassungsdokument erstellen

Verwende folgendes Markdown-Template:

```markdown
# [Fachbereich] - Vorlesungszusammenfassung

**Stand**: [Datum der Analyse]  
**Anzahl analysierter Vorlesungen**: [X]

---

## 📚 Übersicht aller Vorlesungen

| Datum | Hauptthemen | Aufgaben/Termine |
|-------|-------------|------------------|
| 9.10.2025 | Index-Strukturen, Performance | - |
| 16.10.2025 | Hash-Joins, Verbundstrategien | Einsendeaufgabe 2 erwähnt |
| ... | ... | ... |

---

## 📅 Vorlesung vom [Datum]

### 🎯 Kernthemen
- Thema 1
- Thema 2
- Thema 3

### 📝 Wichtige Inhalte

#### [Konzept 1]
- Stichpunkt 1
- Stichpunkt 2
- Beispiel: ...

#### [Konzept 2]
- Stichpunkt 1
- Stichpunkt 2

### ⚠️ AUFGABEN UND TERMINE

> **[Falls vorhanden - sonst Abschnitt weglassen]**

#### Hausaufgabe: [Name/Nummer]
- **Beschreibung**: [Was ist zu tun]
- **Abgabetermin**: [Datum]
- **Hinweise**: 
  - Punkt 1
  - Punkt 2
- **Bewertung**: [Kriterien]

#### Termine
- **[Typ]**: [Datum] - [Beschreibung]

### 💬 Diskussionen
- Frage: [Studentenfrage]
  - Antwort: [Dozentantwort]

### 🔜 Ausblick
- Nächstes Thema: ...
- Skriptkapitel: ...

---

[Wiederholen für jede Vorlesung]

---

## 📋 Gesamtübersicht: Alle Aufgaben und Termine

### Hausaufgaben
1. **Aufgabe [Nr]** - [Kurzbeschreibung]
   - Abgabe: [Datum]
   - Status: [offen/abgeschlossen]

### Wichtige Termine
- **[Datum]**: [Ereignis]
- **[Datum]**: [Ereignis]

---

## 🔍 Themenindex

- **B-Bäume**: Vorlesung vom 9.10.2025, 16.10.2025
- **Hash-Joins**: Vorlesung vom 16.10.2025
- **Indizes**: Vorlesung vom 9.10.2025, 23.10.2025
- ...

```

### Schritt 4: Qualitätssicherung

Prüfe jedes Dokument auf:
- ✅ Sind alle Transkripte erfasst?
- ✅ Sind Termine und Aufgaben vollständig extrahiert?
- ✅ Ist die chronologische Reihenfolge korrekt?
- ✅ Sind die Kernthemen prägnant formuliert?
- ✅ Ist die Übersichtstabelle vollständig?
- ✅ Ist der Themenindex hilfreich?

## Spezielle Hinweise

### Umgang mit unklaren Transkripten
- Wenn Teile unverständlich sind: `[?]` markieren
- Wenn Zahlen/Daten unklar: mit `(ca.)` kennzeichnen
- Wenn Fachbegriffe falsch transkribiert: in eckigen Klammern korrigieren

### Priorisierung
1. **HÖCHSTE PRIORITÄT**: Aufgaben, Abgabetermine, Bewertungshinweise
2. **HOHE PRIORITÄT**: Kernthemen und Hauptkonzepte
3. **MITTLERE PRIORITÄT**: Detaillierte Erklärungen, Beispiele
4. **NIEDRIGE PRIORITÄT**: Allgemeine Diskussionen, organisatorisches

### Kontinuierliche Aktualisierung
Wenn neue Transkripte hinzukommen:
1. Analysiere nur die neuen `.mp4.txt` Dateien
2. Füge die Analyse am Ende des bestehenden Dokuments hinzu
3. Aktualisiere die Übersichtstabelle
4. Aktualisiere den Themenindex
5. Aktualisiere die Gesamtübersicht der Aufgaben/Termine
6. Aktualisiere das Stand-Datum

## Ausführungsbefehl

### Vollständiger Analyse-Prompt (für neue LLM-Session)

```
Ich möchte die Vorlesungstranskripte im Ordner "[ORDNERNAME]/" analysieren.

WICHTIG: Inkrementelle Analyse (nur neue Transkripte verarbeiten)!

Vorgehen:
1. Prüfe, ob die Zusammenfassungsdatei "[ORDNERNAME]_Zusammenfassung.md" bereits existiert
   
2a. FALLS DATEI EXISTIERT:
   - Liste alle .mp4.txt Dateien im Ordner auf
   - Lese die bestehende Zusammenfassungsdatei
   - Identifiziere, welche Vorlesungen bereits analysiert wurden (Datum im Dateinamen)
   - Ermittle die NEUEN Transkripte (noch nicht in Zusammenfassung vorhanden)
   - Analysiere NUR die neuen .mp4.txt Dateien
   - ERGÄNZE die Analysen am Ende der bestehenden Datei
   - Aktualisiere die Übersichtstabelle (füge neue Zeilen hinzu)
   - Aktualisiere den Themenindex (füge neue Einträge hinzu)
   - Aktualisiere "Gesamtübersicht: Aufgaben und Termine"
   - Aktualisiere das "Stand"-Datum im Header
   
2b. FALLS DATEI NICHT EXISTIERT:
   - Liste alle .mp4.txt Dateien im Ordner auf
   - Analysiere ALLE Transkripte von Anfang an
   - Erstelle die Zusammenfassungsdatei komplett neu
   - Sortiere chronologisch (älteste zuerst)

3. Achte BESONDERS auf:
   - Einsendeaufgaben und deren Anforderungen
   - Abgabetermine
   - Bewertungskriterien
   - Hinweise zur Bearbeitung
   - Neue Termine in neu hinzugekommenen Vorlesungen

4. Formatierung:
   - Halte dich strikt an das Markdown-Template
   - Beginne mit der ältesten Vorlesung
   - Arbeite dich chronologisch vor

Analysiere gemäß TRANSCRIPT_ANALYSIS_INSTRUCTIONS.md
```

## Beispiel-Prompts

### Beispiel 1: Erste Analyse (Datei existiert noch nicht)

```
Analysiere alle Transkripte im Ordner "datenbanken/" gemäß 
TRANSCRIPT_ANALYSIS_INSTRUCTIONS.md.

Die Zusammenfassungsdatei "Datenbanken_Zusammenfassung.md" existiert 
noch nicht, daher analysiere ALLE vorhandenen Transkripte und erstelle 
die Datei neu.

Achte besonders auf Aufgaben und Termine!
```

### Beispiel 2: Update (Datei existiert bereits, neue Transkripte hinzugekommen)

```
Analysiere neue Transkripte im Ordner "mad/" gemäß 
TRANSCRIPT_ANALYSIS_INSTRUCTIONS.md.

Die Zusammenfassungsdatei "MAD_Zusammenfassung.md" existiert bereits.

Vorgehen:
1. Liste alle .mp4.txt Dateien im Ordner auf
2. Lese die bestehende Zusammenfassung
3. Ermittle welche Vorlesungen bereits analysiert wurden
4. Analysiere NUR die neuen Transkripte
5. ERGÄNZE die Analysen (nicht überschreiben!)
6. Aktualisiere Übersichtstabelle, Index und Stand-Datum

Achte besonders auf neue Aufgaben und Termine!
```

### Beispiel 3: Prüfung ob Update nötig ist

```
Prüfe den Ordner "pqm/" auf neue Transkripte.

1. Liste alle .mp4.txt Dateien auf
2. Prüfe, ob "PQM_Zusammenfassung.md" existiert
3. Falls ja: Vergleiche welche Transkripte bereits analysiert wurden
4. Falls neue Transkripte vorhanden: Analysiere diese gemäß 
   TRANSCRIPT_ANALYSIS_INSTRUCTIONS.md und ergänze die Zusammenfassung
5. Falls keine neuen Transkripte: Melde "Alle Transkripte bereits analysiert"
```

## Hinweise für den LLM

- **Sei präzise bei Terminen**: Extrahiere exakte Daten, nicht relative Zeitangaben
- **Sei vollständig bei Aufgaben**: Erfasse alle Details zu Hausaufgaben
- **Sei strukturiert**: Halte dich strikt an das Template
- **Sei konsistent**: Verwende einheitliche Formatierung über alle Vorlesungen hinweg
- **Sei kritisch**: Wenn etwas unklar ist, markiere es deutlich

---

**Version**: 1.0  
**Erstellt**: 2025-11-12  
**Für**: Analyse von Vorlesungstranskripten (MAD, PQM, Datenbanken)
