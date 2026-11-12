#!/bin/bash
# Cria o ambiente do projeto no Apuana, como manda o guia do cluster: módulo Python + venv.
#
# Rodar UMA vez, num nó de instalação (o nó de login tem só 8 GB de RAM e é lento):
#   salloc -p install --qos=simple -c 8 --mem=32G --time=00:30:00
#   cd ~/macdeg_classifier && bash cluster/setup_apuana.sh            # ou: ... setup_apuana.sh cu130 Python/3.12.3-GCCcore-13.3.0
# Argumentos opcionais: 1) versão de CUDA das rodas do PyTorch (driver 590 no Apuana -> cu130);
#                       2) módulo Python (padrão: o Python 3 mais novo disponível, de preferência >= 3.11).
# Progresso: tail -f ~/envs/macdeg/install.log (em outra sessão).
set -eo pipefail
CUDA=${1:-cu130}
VENV=${MACDEG_VENV:-$HOME/envs/macdeg}

type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
if [[ "$(hostname)" == slurm-client* ]]; then
  echo "AVISO: você está no nó de login. Rode antes: salloc -p install --qos=simple -c 8 --mem=32G --time=00:30:00"
  read -r -p "Continuar mesmo assim? [s/N] " ok; [[ "$ok" == s* ]] || exit 1
fi

MODULO=${2:-$(module -t avail Python 2>&1 | grep -E '^Python/3\.' | sort -V | tail -1)}
if [ -z "$MODULO" ]; then echo "Nenhum módulo Python encontrado (module avail Python)."; exit 1; fi
echo "Módulo Python: $MODULO"
module load "$MODULO"
python3 -c "import sys; assert sys.version_info >= (3, 10), sys.version; print('Python', sys.version.split()[0])"

if [ -d "$VENV" ]; then
  echo "Ambiente $VENV já existe: apague-o (rm -rf $VENV) para recriar do zero, ou ele só será atualizado."
else
  python3 -m venv "$VENV"
fi
echo "$MODULO" > "$VENV/MODULO_PYTHON"
source "$VENV/bin/activate"
LOG="$VENV/install.log"
echo "== $(date) instalação em $(hostname)" > "$LOG"

echo "PyTorch ($CUDA, ~3 GB). Progresso: tail -f $LOG"
pip install --upgrade pip >> "$LOG" 2>&1
pip install --progress-bar on torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}" 2>&1 | tee -a "$LOG"
echo "Demais pacotes..."
pip install --progress-bar on -r cluster/requirements-cluster.txt 2>&1 | tee -a "$LOG"
pip freeze > "$VENV/freeze.txt"

# Pesos da ResNet-50 baixados agora: os nós de GPU podem não ter internet
python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V2)"
# Kernel do Jupyter apontando para este ambiente (o papermill executa os notebooks com ele)
python -m ipykernel install --user --name python3 --display-name "Python 3 (macdeg)" >/dev/null
chmod 700 ~/macdeg_data 2>/dev/null || true
python -c "import torch, pandas, papermill, wandb; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| pandas', pandas.__version__, '| ok')"
echo "Ambiente pronto em $VENV (módulo $MODULO). Versões: $VENV/freeze.txt"
