# DeepCoder Pipeline

Diese Pipeline reproduziert und wertet die Ergebnisse des DeepCoder-Solvers
(DFS-Suche mit optionalem neuronalem Prädiktor) auf dem `T=2`-Testset aus.
Sie läuft komplett nativ unter Windows (kein WSL nötig), über das
`.venv` (`C:\BA\deepcoder\.venv`, Python 3.9 + TensorFlow/Keras 2.12).

## Schnellstart: alles mit einem Befehl

`run_all.sh` führt die komplette Pipeline (DFS-Baseline → Training →
DFS+Prädiktor → CSV-Export → Metriken → Notebook) in einem Rutsch aus, per
Git Bash:

```bash
cd /c/BA/deepcoder/deepcoder_pipeline
./run_all.sh
```

Konfigurierbar über Umgebungsvariablen (Defaults in Klammern):

| Variable | Bedeutung | Default |
|---|---|---|
| `T` | Aufgabenkomplexität (Datasetauswahl `T=<T>_train/test.json`) | `2` |
| `GAS` | DFS-Suchbudget | `10000` |
| `EPOCHS` | Trainingsepochen für den Prädiktor | `10` |
| `SEED` | Zufalls-Seed für das Training (reproduzierbare Gewichtsinitialisierung) | `42` |
| `DEEPCODER_REPO_ROOT` | Pfad zum `deepcoder`-Repo (für `PYTHONPATH` + Dataset-Verzeichnis) | Parent von `deepcoder_pipeline/` |
| `PYTHON` | Python-Interpreter | `.venv/Scripts/python.exe` im Repo |

Beispiel für einen zweiten Lauf mit anderem Budget:

```bash
GAS=1000 ./run_all.sh
```

Der in der Arbeit zitierte Lauf verwendet tatsächlich `GAS=1000` (nicht den Default `10000`), dafür über 5 Seeds gemittelt statt nur `seed=42`. Siehe Abschnitt "Multi-Seed-Lauf" und "Ergebnisse" unten für die genaue Konfiguration.

Jeder Lauf (jede `T`/`GAS`/`EPOCHS`/`SEED`-Kombination) bekommt einen
eigenen Ordner `output/runs/T<T>_gas<GAS>_epochs<EPOCHS>_seed<SEED>/` mit
einer Datei pro Pipeline-Schritt und einem `logs/`-Unterordner darin —
nichts wird zwischen unterschiedlichen Konfigurationen oder Schritten
überschrieben. Das Notebook liest `T`/`GAS`/`EPOCHS`/`SEED` aus denselben
Umgebungsvariablen (Default wie im Notebook selbst: `T=2`, `GAS=1000`,
`EPOCHS=10`, `SEED=42`) und schreibt in denselben Run-Ordner wie
`run_all.sh`.

### Reproduzierbarkeit

`train-nn.py` setzt jetzt per `--seed` (Default `42`) einen festen Zufalls-Seed
für `random`, `numpy` und `tensorflow`. Ohne das lieferte jeder Trainingslauf
eine andere Gewichtsinitialisierung und damit eine andere Accuracy nach dem
Training (beobachtet: 65-73% bei sonst identischer Konfiguration, gas=1000).
Mit festem Seed sind zwei Läufe mit gleichem `T`/`GAS`/`EPOCHS`/`SEED`
bit-identisch in allen Kennzahlen (getestet: zwei Notebook-Läufe hintereinander
ergaben exakt 65/100 gelöst, mean nb_steps 508.9, 19% Exact Match,
partial_correctness 0.4097 — beide Male identisch). Die DFS-Baseline ohne
Training ist ohnehin deterministisch (keine Zufallskomponente).

## Ordnerstruktur

```
deepcoder_pipeline/
├── run_all.sh                            Ein Befehl für die komplette Pipeline
├── scripts/                              Kopien der ausführbaren Pipeline-Schritte
│   ├── solve-problems.py                 DFS-Solver (mit/ohne Prädiktor)
│   ├── train-nn.py                       Training des neuronalen Prädiktors
│   ├── export_results_to_csv.py          Konvertiert .h5-Ergebnisse nach CSV
│   ├── compute_metrics.py                CLI-Variante der Notebook-Metriken (+ partial_correctness, best_effort_*)
│   ├── plot_multiseed.py                 Regeneriert die Haupt-Vergleichsdiagramme über alle 5 Seeds (Thesis-Metriknamen)
│   └── plot_unsolved.py                  Regeneriert die beiden Best-Effort-Diagramme über alle 5 Seeds
├── output/
│   └── runs/                              ein Ordner pro T/GAS/EPOCHS/SEED-Kombination
│       ├── T2_gas1000_epochs10_seed{0,1,2,3,4}/   die 5 Seeds des in der Arbeit zitierten Laufs
│       └── T2_gas10000_epochs10_seed42/           Schnellstart-Beispiel unten (einzelner Seed, größeres Budget)
│           ├── model.h5                       trainierter Prädiktor
│           ├── solve_no_predictor.h5/.csv     DFS ohne Training
│           ├── solve_with_predictor.h5/.csv   DFS mit Training
│           ├── metrics_no_predictor.csv       compute_metrics.py-Output (siehe unten)
│           ├── metrics_with_predictor.csv     compute_metrics.py-Output (siehe unten)
│           ├── metric_comparison.png          Notebook-Diagramm
│           ├── pipeline_summary.txt           Notebook-Zusammenfassung
│           └── logs/                          ein Log pro Pipeline-Schritt
├── multiseed_run.log                      Gesamtlog des 5-Seed-Laufs, der für die Arbeit zitiert wird
├── deepcoder_metric_analysis.ipynb        Metrik-Auswertung + Diagramme
└── README.md                              diese Datei
```

Jedes `solve_*`-Ergebnis existiert als `.h5` (Rohausgabe von
`solve-problems.py`, inkl. `reference`/`solution`/`nb_steps`/`wall_ms`) und
als `.csv` (nach `solved`-Spalte aufbereitet, für das Notebook).

### Hinweis: Worker-Anzahl in `solve-problems.py`

Unsere Kopie von `solve-problems.py` begrenzt die `ProcessPoolExecutor`
auf `min(4, os.cpu_count())` statt (wie im Original) alle CPU-Kerne zu
nutzen. Grund: unter Windows importiert jeder Worker-Prozess das komplette
Skriptmodul neu (`spawn` statt `fork`), inklusive der schweren
`keras`/`tensorflow`/`jax`-Importkette. Mit 8 Kernen und wenig freiem RAM
(z. B. während parallel ein Jupyter-Kernel läuft) kann das zu
`MemoryError` in einzelnen Workern und damit zu einem `BrokenProcessPool`
führen. 4 Worker sind ein guter Kompromiss aus Laufzeit und Speicherbedarf.

## Warum diese Pipeline nötig war

Der ursprüngliche DeepCoder-Fork hatte zwei Bugs, die die Ergebnisse stark
verfälscht haben (behoben in `deepcoder/dsl/value.py`,
`deepcoder/nn/model.py`, `deepcoder/nn/encoding.py`):

1. **Tuple/List-Vergleichsbug**: `MAP`/`FILTER`/`ZIPWITH`/`SCAN1L` lieferten
   Tupel zurück, während aus JSON geladene Beispiel-Outputs Listen waren.
   `[1, 2] != (1, 2)` in Python führte dazu, dass strukturell korrekte
   Programme als "nicht gelöst" gewertet wurden. Deckelte die Accuracy
   künstlich auf ~17-23%.
2. **Falsche Loss-Funktion für den Prädiktor**: Die Attribut-Labels sind
   Multi-Hot (mehrere DSL-Funktionen können gleichzeitig im Zielprogramm
   vorkommen), das Modell nutzte aber `softmax` + `categorical_crossentropy`
   (Single-Label). Das Training divergierte (Loss stieg statt zu fallen),
   der Prädiktor machte die Suche schlechter statt besser. Fix: `sigmoid` +
   `binary_crossentropy`.

Nach dem Fix reproduziert die DFS-Baseline exakt die im Original-README
dokumentierten 53% (gas=1000), und das Training verbessert die Ergebnisse
wieder wie erwartet.

## Pipeline-Schritte

Alle Befehle aus `deepcoder_pipeline/` heraus, mit
`PYTHONPATH=C:\BA\deepcoder` gesetzt (damit `import deepcoder` auflöst).
`RUN` steht für den Run-Ordner der jeweiligen Hyperparameter-Kombination
(`output\runs\T2_gas10000_epochs10_seed42`):

```powershell
cd C:\BA\deepcoder\deepcoder_pipeline
$env:PYTHONPATH = "C:\BA\deepcoder"
$RUN = "output\runs\T2_gas10000_epochs10_seed42"
New-Item -ItemType Directory -Force "$RUN\logs" | Out-Null
```

### 1. DFS-Solver ohne Prädiktor

```powershell
..\.venv\Scripts\python.exe scripts\solve-problems.py `
    ..\dataset\T=2_test.json --T 2 --mode dfs --gas 10000 `
    --outfile "$RUN\solve_no_predictor.h5"
```

### 2. Prädiktor trainieren

```powershell
..\.venv\Scripts\python.exe scripts\train-nn.py `
    --in ..\dataset\T=2_train.json --out "$RUN\model.h5" --epochs 10 --seed 42
```

### 3. DFS-Solver mit Prädiktor

```powershell
..\.venv\Scripts\python.exe scripts\solve-problems.py `
    ..\dataset\T=2_test.json --T 2 --mode dfs --gas 10000 `
    --predictor "$RUN\model.h5" `
    --outfile "$RUN\solve_with_predictor.h5"
```

### 4. Ergebnisse nach CSV exportieren

```powershell
..\.venv\Scripts\python.exe scripts\export_results_to_csv.py `
    --infile "$RUN\solve_no_predictor.h5" `
    --outfile "$RUN\solve_no_predictor.csv"
```

(analog für die andere `.h5`-Datei aus Schritt 3)

### 5. Notebook: die Pipeline Schritt für Schritt (wie bei DreamCoder)

`deepcoder_metric_analysis.ipynb` ist — inspiriert von
`dreamcoder_train_test_pipeline_step_by_step/DreamCoder_Train_Test_Workflow.ipynb`
— kein reines Analyse-Notebook mehr, sondern orchestriert die komplette
Pipeline selbst, eine Zelle pro Schritt:

1. **Konfiguration** (`T`, `GAS`, `EPOCHS`, `SEED` am Anfang — liest sie aus
   Umgebungsvariablen, falls gesetzt, sonst Defaults `2`/`1000`/`10`/`42`;
   baut daraus denselben Run-Ordner wie `run_all.sh`:
   `output/runs/T<T>_gas<GAS>_epochs<EPOCHS>_seed<SEED>/`)
2. `run_step()`-Hilfsfunktion, die jedes `scripts/*.py` als Subprozess
   startet, den Output live ins Notebook streamt und nach `RUN_DIR/logs/`
   schreibt (bricht mit Fehler ab, falls ein Schritt fehlschlägt)
3. Schritt 1 – DFS ohne Prädiktor, Schritt 2 – Training, Schritt 3 – DFS mit
   Prädiktor, Schritt 4 – CSV-Export, Schritt 5 – `compute_metrics.py` für
   beide Konfigurationen
4. Vergleichstabelle + gruppiertes Balkendiagramm (ohne vs. mit Training,
   mit Wertbeschriftung über jedem Balken) — gespeichert als
   `RUN_DIR/metric_comparison.png`
5. Finale Textzusammenfassung, gespeichert als
   `RUN_DIR/pipeline_summary.txt`

Ausführen:

```powershell
..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    deepcoder_metric_analysis.ipynb
```

Andere Hyperparameter testen: `T`/`GAS`/`EPOCHS`/`SEED` in der
Konfigurationszelle (oder als Umgebungsvariable vor dem Start) ändern und
erneut ausführen — jeder Lauf bekommt einen eigenen Run-Ordner, nichts wird
überschrieben.

### 6. Metriken per Skript (von Schritt 5 im Notebook aufgerufen, auch einzeln nutzbar)

```powershell
..\.venv\Scripts\python.exe scripts\compute_metrics.py `
    --infile "$RUN\solve_no_predictor.csv" `
    --outfile "$RUN\metrics_no_predictor.csv"
```

Berechnet dieselben vier Ähnlichkeitsmetriken wie das Notebook
(`program_operation_score`, `program_position_score`,
`program_sequence_score`, `program_edit_score`), plus:

- `exact_match`: **exakte Token-Gleichheit** von Referenz- und gefundenem
  Programm (strukturell), nicht dasselbe wie `solved` (hat die I/O-Beispiele
  erfüllt, funktional). Auf dem 5-Seed-Lauf ohne Training ergibt das nur
  11/100 exakte Treffer gegenüber 53/100 funktional gelösten Aufgaben —
  DeepCoder findet meistens ein alternatives, aber ebenso korrektes Programm.
- `partial_correctness`: gleich gewichteter Durchschnitt der vier
  Ähnlichkeitsmetriken.
- `best_effort_operation_score`/`_position_score`/`_sequence_score`/`_edit_score`/`_partial_correctness`:
  dieselben vier Metriken, aber für unsolved Aufgaben statt hart mit `0`
  bewertet, gegen `best_partial_solution` bewertet, den besten während der
  DFS-Suche gesehenen Kandidaten (siehe `deepcoder/search.py`), falls einer
  existiert. `best_effort_program` ist dabei die Lösung, wenn gelöst, sonst
  dieser Kandidat, sonst leer.

Vor dem Vergleich wird bei allen vier Metriken zusätzlich das führende
Typsignatur-Präfix (`LIST`, `INT`, …) aus Referenz- und Kandidatenprogramm
entfernt (`strip_type_markers`); da Referenz und Kandidat für eine Aufgabe
immer denselben Typ-Präfix haben, trägt dieser sonst keine Information über
die Suchqualität, sondern würde die Werte künstlich nach oben verzerren.

## Multi-Seed-Lauf (der in der Arbeit zitierte Lauf)

Die DFS-Baseline ohne Training ist deterministisch (kein Zufallsanteil), das
trainierte Netz dagegen nicht: unterschiedliche Gewichtsinitialisierungen
führen zu unterschiedlichen Solve-Raten. Für die Arbeit wird deshalb nicht
ein einzelner Seed berichtet, sondern Mittelwert ± Standardabweichung über
`SEED=0,1,2,3,4` bei `T=2`, `GAS=1000`, `EPOCHS=10`:

```bash
cd /c/BA/deepcoder/deepcoder_pipeline
for SEED in 0 1 2 3 4; do
  T=2 GAS=1000 EPOCHS=10 SEED=$SEED ./run_all.sh
done
```

Jeder Seed bekommt seinen eigenen Run-Ordner
(`output/runs/T2_gas1000_epochs10_seed<N>/`), nichts wird überschrieben.
Anschließend erzeugen die beiden Diagramm-Skripte die für die Arbeit
verwendeten Diagramme direkt aus diesen fünf Run-Ordnern, mit der Kurzform
der Metriken, die auch die Arbeit verwendet (BSS/POS/PPS/PSS/PES):

```bash
../.venv/Scripts/python.exe scripts/plot_multiseed.py
../.venv/Scripts/python.exe scripts/plot_unsolved.py
```

`plot_multiseed.py` fasst die "mit Training"-Konfiguration zu einem
Mittelwert-±-Std-Balken über die fünf Seeds zusammen. `plot_unsolved.py`
macht dasselbe für die beiden Best-Effort-Diagramme (siehe unten), einmal
über alle ungelösten Aufgaben und einmal beschränkt auf die, die tatsächlich
einen gespeicherten Best-Effort-Kandidaten haben.

## Ergebnisse (T=2 Testset, 100 Aufgaben, gas=1000, 5 Seeds)

| Konfiguration | gelöst (funktional) | exakte Programmübereinstimmung |
|---|---|---|
| ohne Training | 53/100 (53,0 %) | 11/100 (11,0 %) |
| mit Training, Mittel ± Std | 67,8/100 (67,8 % ± 1,8 %) | 21,0 % ± 1,3 % |

Einzelne Seeds mit Training lösen zwischen 66 und 70 der 100 Aufgaben.
"Exakte Programmübereinstimmung" ist strenger als "gelöst" (Token-für-Token
identisch zum Referenzprogramm statt nur funktional äquivalent) — der
deutlich größere Abstand zu den 53 %/67,8 % oben zeigt, dass DeepCoder in
den meisten gelösten Fällen ein alternatives, aber ebenso korrektes
Programm findet statt exakt des Referenzprogramms.

Die vier strukturellen Ähnlichkeitsmetriken aus `compute_metrics.py`, je über
alle 100 Testaufgaben und über nur die gelösten:

| Konfiguration | POS (alle / gelöst) | PPS (alle / gelöst) | PSS (alle / gelöst) | PES (alle / gelöst) |
|---|---|---|---|---|
| ohne Training | 0,180 / 0,340 | 0,145 / 0,274 | 0,180 / 0,340 | 0,145 / 0,274 |
| mit Training, Mittel ± Std | 0,274±0,014 / 0,404±0,011 | 0,268±0,017 / 0,395±0,016 | 0,274±0,014 / 0,404±0,011 | 0,268±0,017 / 0,395±0,016 |

POS und PSS sind hier identisch (ebenso PPS und PES): Bei den zweistufigen
Referenzprogrammen dieses Datensatzes fällt bei DeepCoder jede Token-Menge,
die der POS als gemeinsam zählt, auch als zusammenhängender Block auf, und
jede exakte Positionsübereinstimmung kostet auch beim Editierabstand genau
eine Ersetzung.

### Best-Effort-Kandidaten (nur ungelöste Aufgaben)

`best_effort_*` bewertet ungelöste Aufgaben gegen den besten während der
DFS-Suche gesehenen Teilkandidaten statt sie pauschal mit `0` zu bewerten.
Nicht jede ungelöste Aufgabe hat einen gespeicherten Kandidaten, deshalb
zwei Sichten:

| Konfiguration | POS über alle ungelösten | POS nur mit Kandidat | PPS/PES über alle ungelösten | PPS/PES nur mit Kandidat |
|---|---|---|---|---|
| ohne Training (47 ungelöst, 25 mit Kandidat) | 0,021 | 0,040 | ≈0,000 | ≈0,000 |
| mit Training, Mittel ± Std (Ø 32 ungelöst, Ø 20 mit Kandidat) | 0,043±0,011 | 0,069±0,016 | 0,003±0,006 | 0,005±0,010 |

Selbst ein gespeicherter Best-Effort-Kandidat, der speziell deshalb behalten
wurde, weil er mehr der gegebenen Beispiele erfüllte als jeder andere von
der Suche ausprobierte Kandidat, ist strukturell also kaum näher an der
Referenz als eine ungelöste Aufgabe ohne jeden Kandidaten.
