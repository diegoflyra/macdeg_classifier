#!/bin/bash
# Um experimento = um job = uma GPU. Roda as tarefas uma por vez, a análise e o envio dos resultados.
#
# Do nó de login, na raiz do repositório (crie a pasta de logs uma vez: mkdir -p logs):
#   sbatch cluster/experimento.sh                                        # a referência
#   sbatch cluster/experimento.sh configs/experimentos/002_x.yaml        # outro experimento
#   sbatch -p debug --qos=complex --time=00:30:00 cluster/experimento.sh --teste       # teste rápido
#   sbatch cluster/experimento.sh --retomar ~/macdeg_runs/<lote>          # continua um lote interrompido
#   sbatch cluster/experimento.sh --final ~/macdeg_runs/<lote> <configuração>   # modelo final -> HF
# Os argumentos vão direto para o rodar.py. Saída em logs/macdeg_<job>.out, erros em logs/macdeg_<job>.err.
#SBATCH --job-name=macdeg
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "=========================================="
echo "job $SLURM_JOB_ID | nó $(hostname) | $(date '+%Y-%m-%d %H:%M:%S')"
echo "argumentos: $*"
echo "=========================================="

# 1. Ambiente
for v in .venv venv; do
  if [ -f "$v/bin/activate" ]; then source "$v/bin/activate"; break; fi
done
command -v python >/dev/null || { echo "ERRO: venv não encontrado (.venv ou venv na raiz do repositório)" >&2; exit 1; }
echo "python: $(which python)"
nvidia-smi -L || echo "AVISO: nvidia-smi falhou" >&2
python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponível'; print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))"

# 2. Pastas
export MACDEG_RUNS=${MACDEG_RUNS:-$HOME/macdeg_runs}
mkdir -p "$MACDEG_RUNS" logs

# 3. Dados no disco local do nó (rápido); apagados no fim, mesmo se o job falhar ou for cancelado
TMP="${SLURM_TMPDIR:-/tmp}/$USER-macdeg-$SLURM_JOB_ID"
trap 'rm -rf "$TMP"' EXIT TERM INT
mkdir -p "$TMP"
echo "copiando o conjunto para $TMP ..."
python -m zipfile -e "$HOME/macdeg_data/dataset_1024.zip" "$TMP/dataset"
echo "conjunto pronto: $(ls "$TMP/dataset/images" | wc -l) imagens"

# 4. Experimento (tarefas, análise, GitHub, W&B; HF no --final)
set +e
python rodar.py --dados "$TMP/dataset" "$@"
STATUS=$?
set -e

# 5. Logs do job vão junto com o lote (e para o GitHub, exceto no teste)
LOTE=$(cat "$MACDEG_RUNS/ultimo_lote.txt" 2>/dev/null || true)
if [ -n "$LOTE" ] && [ -d "$LOTE" ]; then
  cp "logs/macdeg_${SLURM_JOB_ID}.out" "logs/macdeg_${SLURM_JOB_ID}.err" "$LOTE/logs/" 2>/dev/null || true
  python -m macdeg.analysis logs "$LOTE" || true
fi
echo "fim | status $STATUS | $(date '+%Y-%m-%d %H:%M:%S')"
exit $STATUS
