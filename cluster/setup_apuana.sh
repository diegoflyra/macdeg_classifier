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

# Python: o módulo mais novo, mas só se for mais novo que o Python do sistema (que existe igual em todos
# os nós). Em 2026-11 o Apuana tem módulos até 3.10.8 e o sistema tem 3.10.12: usa-se o do sistema.
SISTEMA=$(/usr/bin/python3 -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
MODULO=${2:-$(module -t avail Python 2>&1 | grep -E '^Python/3\.' | sort -V | tail -1)}
VMOD=$(echo "$MODULO" | sed -E 's|Python/([0-9.]+).*|\1|')
if [ -n "$MODULO" ] && { [ -n "${2:-}" ] || [ "$(printf '%s\n%s\n' "$SISTEMA" "$VMOD" | sort -V | tail -1)" != "$SISTEMA" ]; }; then
  echo "Python do módulo $MODULO (sistema: $SISTEMA)"
  module load "$MODULO"
  PY=python3
else
  echo "Python do sistema $SISTEMA (módulo mais novo: ${MODULO:-nenhum}, não é mais novo)"
  MODULO=""
  PY=/usr/bin/python3
fi
$PY -c "import sys; assert sys.version_info >= (3, 10), sys.version; print('Python', sys.version.split()[0])"

if [ -d "$VENV" ]; then
  echo "Ambiente $VENV já existe: apague-o (rm -rf $VENV) para recriar do zero, ou ele só será atualizado."
else
  $PY -m venv "$VENV"
fi
if [ -n "$MODULO" ]; then echo "$MODULO" > "$VENV/MODULO_PYTHON"; else rm -f "$VENV/MODULO_PYTHON"; fi
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
echo "Ambiente pronto em $VENV (${MODULO:-Python do sistema}). Versões: $VENV/freeze.txt"
