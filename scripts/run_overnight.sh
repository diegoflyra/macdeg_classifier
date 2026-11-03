#!/usr/bin/env bash
# Pré-processamento, grades, auditoria e testes. Retoma de onde parou se for rodado de novo.
# Uso, na raiz do projeto: bash scripts/run_overnight.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
LOG="outputs/logs/overnight_$(date +%Y%m%d_%H%M%S).log"
PY=.venv/bin/python

run() {
  echo "=== $1: início $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  echo "=== fim $(date '+%H:%M:%S'), código $status ===" | tee -a "$LOG"
  return "$status"
}

run "Pré-processamento" "$PY" -m macdeg.preprocess --workers 6 || { echo "Parou no pré-processamento. Veja $LOG"; exit 1; }
run "Grades" "$PY" -m macdeg.grids
run "Auditoria" "$PY" -m macdeg.audit --n 200 --workers 6
run "Testes" "$PY" -m pytest tests -q
echo "Concluído. Log: $LOG"
