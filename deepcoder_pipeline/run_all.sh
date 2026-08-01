#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPCODER_REPO_ROOT="${DEEPCODER_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-$DEEPCODER_REPO_ROOT/.venv/Scripts/python.exe}"

T="${T:-2}"
GAS="${GAS:-10000}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
TRAIN_DATASET_FILE="${TRAIN_DATASET_FILE:-dataset/T=${T}_train.json}"
TEST_DATASET_FILE="${TEST_DATASET_FILE:-dataset/T=${T}_test.json}"

TRAIN_PATH="$DEEPCODER_REPO_ROOT/$TRAIN_DATASET_FILE"
TEST_PATH="$DEEPCODER_REPO_ROOT/$TEST_DATASET_FILE"

# Every hyperparameter combination gets its own run directory, so nothing
# from a previous run (results *or* logs) is silently overwritten.
RUN_NAME="T${T}_gas${GAS}_epochs${EPOCHS}_seed${SEED}"
RUN_DIR="$SCRIPT_DIR/output/runs/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$DEEPCODER_REPO_ROOT"
# Passed through so the notebook (step 6) can build the same RUN_DIR.
export T GAS EPOCHS SEED

echo "Pipeline started: $(date)"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "DEEPCODER_REPO_ROOT=$DEEPCODER_REPO_ROOT"
echo "PYTHON=$PYTHON"
echo "T=$T"
echo "GAS=$GAS"
echo "EPOCHS=$EPOCHS"
echo "SEED=$SEED"
echo "TRAIN_PATH=$TRAIN_PATH"
echo "TEST_PATH=$TEST_PATH"
echo "RUN_DIR=$RUN_DIR"
echo

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Missing training dataset: $TRAIN_PATH"
  echo "Set TRAIN_DATASET_FILE (relative to \$DEEPCODER_REPO_ROOT) explicitly if needed."
  exit 1
fi
if [[ ! -f "$TEST_PATH" ]]; then
  echo "Missing test dataset: $TEST_PATH"
  echo "Set TEST_DATASET_FILE (relative to \$DEEPCODER_REPO_ROOT) explicitly if needed."
  exit 1
fi

NO_PRED_H5="$RUN_DIR/solve_no_predictor.h5"
NO_PRED_CSV="$RUN_DIR/solve_no_predictor.csv"
WITH_PRED_H5="$RUN_DIR/solve_with_predictor.h5"
WITH_PRED_CSV="$RUN_DIR/solve_with_predictor.csv"
MODEL_FILE="$RUN_DIR/model.h5"

echo "== step 1/5: DFS-Solver ohne Praediktor (gas=$GAS) =="
"$PYTHON" "$SCRIPT_DIR/scripts/solve-problems.py" "$TEST_PATH" --T "$T" --mode dfs --gas "$GAS" \
  --outfile "$NO_PRED_H5" | tee "$LOG_DIR/step1_solve_no_predictor.log"

echo "== step 2/5: Praediktor trainieren ($EPOCHS Epochen, seed=$SEED) =="
"$PYTHON" "$SCRIPT_DIR/scripts/train-nn.py" --in "$TRAIN_PATH" --out "$MODEL_FILE" --epochs "$EPOCHS" \
  --seed "$SEED" | tee "$LOG_DIR/step2_train_nn.log"

echo "== step 3/5: DFS-Solver mit Praediktor (gas=$GAS) =="
"$PYTHON" "$SCRIPT_DIR/scripts/solve-problems.py" "$TEST_PATH" --T "$T" --mode dfs --gas "$GAS" \
  --predictor "$MODEL_FILE" --outfile "$WITH_PRED_H5" | tee "$LOG_DIR/step3_solve_with_predictor.log"

echo "== step 4/5: Ergebnisse nach CSV exportieren =="
"$PYTHON" "$SCRIPT_DIR/scripts/export_results_to_csv.py" --infile "$NO_PRED_H5" --outfile "$NO_PRED_CSV" \
  | tee "$LOG_DIR/step4a_export_no_predictor.log"
"$PYTHON" "$SCRIPT_DIR/scripts/export_results_to_csv.py" --infile "$WITH_PRED_H5" --outfile "$WITH_PRED_CSV" \
  | tee "$LOG_DIR/step4b_export_with_predictor.log"

echo "== step 5/5: Metriken berechnen (compute_metrics.py) =="
"$PYTHON" "$SCRIPT_DIR/scripts/compute_metrics.py" --infile "$NO_PRED_CSV" \
  --outfile "$RUN_DIR/metrics_no_predictor.csv" \
  | tee "$LOG_DIR/step5a_compute_metrics_no_predictor.log"
"$PYTHON" "$SCRIPT_DIR/scripts/compute_metrics.py" --infile "$WITH_PRED_CSV" \
  --outfile "$RUN_DIR/metrics_with_predictor.csv" \
  | tee "$LOG_DIR/step5b_compute_metrics_with_predictor.log"

echo "== bonus: Notebook ausfuehren (Diagramme + Zusammenfassung, gleicher RUN_DIR) =="
( cd "$SCRIPT_DIR" && "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
    deepcoder_metric_analysis.ipynb ) | tee "$LOG_DIR/step6_notebook.log"

echo
echo "Pipeline finished: $(date)"
echo "Ergebnisse liegen in: $RUN_DIR"
