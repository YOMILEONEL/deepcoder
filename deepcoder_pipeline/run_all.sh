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

OUTPUT_DIR="$SCRIPT_DIR/output"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$DEEPCODER_REPO_ROOT"

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
echo "OUTPUT_DIR=$OUTPUT_DIR"
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

NO_PRED_H5="$OUTPUT_DIR/results_T${T}_gas${GAS}_no_predictor.h5"
NO_PRED_CSV="$OUTPUT_DIR/results_T${T}_gas${GAS}_no_predictor.csv"
WITH_PRED_H5="$OUTPUT_DIR/results_T${T}_gas${GAS}_with_predictor.h5"
WITH_PRED_CSV="$OUTPUT_DIR/results_T${T}_gas${GAS}_with_predictor.csv"
MODEL_FILE="$OUTPUT_DIR/model_T${T}.h5"

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
  | tee "$LOG_DIR/step4_export_no_predictor.log"
"$PYTHON" "$SCRIPT_DIR/scripts/export_results_to_csv.py" --infile "$WITH_PRED_H5" --outfile "$WITH_PRED_CSV" \
  | tee "$LOG_DIR/step4_export_with_predictor.log"

echo "== step 5/5: Metriken berechnen (compute_metrics.py) =="
"$PYTHON" "$SCRIPT_DIR/scripts/compute_metrics.py" --infile "$NO_PRED_CSV" \
  --outfile "$OUTPUT_DIR/results_T${T}_gas${GAS}_no_predictor_compute_metrics.csv" \
  | tee "$LOG_DIR/step5_compute_metrics_no_predictor.log"
"$PYTHON" "$SCRIPT_DIR/scripts/compute_metrics.py" --infile "$WITH_PRED_CSV" \
  --outfile "$OUTPUT_DIR/results_T${T}_gas${GAS}_with_predictor_compute_metrics.csv" \
  | tee "$LOG_DIR/step5_compute_metrics_with_predictor.log"

if [[ "$T" == "2" && "$GAS" == "1000" ]]; then
  echo "== bonus: Notebook ausfuehren (Diagramme, fest auf T=2/gas=1000 verdrahtet) =="
  ( cd "$SCRIPT_DIR" && "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
      deepcoder_metric_analysis.ipynb ) | tee "$LOG_DIR/step6_notebook.log"
else
  echo "== Notebook uebersprungen (fest auf T=2/gas=1000 verdrahtet, aktueller Lauf: T=$T gas=$GAS) =="
fi

echo
echo "Pipeline finished: $(date)"
echo "Ergebnisse liegen in: $OUTPUT_DIR"
