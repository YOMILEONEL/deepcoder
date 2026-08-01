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
│   └── compute_metrics.py                CLI-Variante der Notebook-Metriken (+ partial_correctness)
├── output/
│   └── runs/                              ein Ordner pro T/GAS/EPOCHS/SEED-Kombination
│       └── T2_gas10000_epochs10_seed42/
│           ├── model.h5                       trainierter Prädiktor
│           ├── solve_no_predictor.h5/.csv     DFS ohne Training
│           ├── solve_with_predictor.h5/.csv   DFS mit Training
│           ├── metrics_no_predictor.csv       compute_metrics.py-Output (siehe unten)
│           ├── metrics_with_predictor.csv     compute_metrics.py-Output (siehe unten)
│           ├── metric_comparison.png          Notebook-Diagramm
│           ├── pipeline_summary.txt           Notebook-Zusammenfassung
│           └── logs/                          ein Log pro Pipeline-Schritt
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

Berechnet dieselben vier Ähnlichkeitsmetriken wie das Notebook, aber mit
zwei Unterschieden:

- `accuracy` bedeutet hier **exakte Token-Gleichheit** von Referenz- und
  gefundenem Programm (strukturell), nicht "hat die I/O-Beispiele erfüllt"
  wie im Notebook (`solved`-Spalte, funktional). Auf
  `results_T2_gas10000_no_predictor` ergibt das nur 31/100 exakte Treffer
  gegenüber 93/100 funktional gelösten Aufgaben — DeepCoder findet in den
  meisten Fällen ein alternatives, aber ebenso korrektes Programm.
- zusätzlich wird `partial_correctness` berechnet (gleich gewichteter
  Durchschnitt der vier Ähnlichkeitsmetriken).

## Ergebnisse (T=2 Testset, 100 Aufgaben)

| Konfiguration | gas | gelöst (funktional) | Ø nb_steps |
|---|---|---|---|
| ohne Training | 1000 | 53/100 (53%) | 628 |
| mit Training | 1000 | 73/100 (73%) | 446 |
| ohne Training | 10000 | 93/100 (93%) | 2398 |
| mit Training | 10000 | 100/100 (100%) | 998 |

Bei gleichem Suchbudget (`gas`) verbessert das trainierte Netz sowohl die
Trefferquote als auch die Effizienz der Suche deutlich (weniger
`nb_steps` im Mittel). Ein größeres `gas`-Budget allein (ohne Training)
verbessert ebenfalls die Trefferquote, ist aber kein Ersatz für den
Prädiktor — mit Training + gas=10000 werden alle 100 Aufgaben gelöst, bei
weniger als der Hälfte der durchschnittlichen Suchschritte.

Zusätzlich, aus `compute_metrics.py` (gas=10000):

| Konfiguration | exakte Programmübereinstimmung | partial_correctness (Ø) |
|---|---|---|
| ohne Training | 31/100 (31%) | 0.6256 |
| mit Training | 42/100 (42%) | 0.7225 |

"Exakte Programmübereinstimmung" ist strenger als "gelöst" (Token-für-Token
identisch zum Referenzprogramm statt nur funktional äquivalent) — der große
Abstand zu den 93%/100% oben zeigt, dass DeepCoder meist ein alternatives,
aber ebenso korrektes Programm findet statt exakt des Referenzprogramms.
