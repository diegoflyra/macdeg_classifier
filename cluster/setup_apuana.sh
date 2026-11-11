#!/bin/bash
# Prepara o ambiente no Apuana. Rodar UMA vez, no nó de login, de dentro do clone do repositório:
#   bash cluster/setup_apuana.sh [cu130]
# O argumento é a versão de CUDA das rodas do PyTorch, conforme o driver (nvidia-smi num nó de GPU):
# driver >= 580 -> cu130 (padrão; o Apuana tem 590 em 2026-11); 550-579 -> cu124; 525-549 -> cu121.
# O Apuana não tem conda no login: instalamos o Miniforge no $HOME, que é compartilhado com os nós.
set -euo pipefail
CUDA=${1:-cu130}
BASE=${MACDEG_CONDA:-$HOME/miniforge3}
if [ ! -f "$BASE/etc/profile.d/conda.sh" ]; then
  echo "Instalando Miniforge em $BASE ..."
  curl -fsSL -o /tmp/miniforge_$USER.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge_$USER.sh -b -p "$BASE"
  rm -f /tmp/miniforge_$USER.sh
fi
source "$BASE/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^macdeg "; then
  conda create -y -n macdeg python=3.12
fi
conda activate macdeg
pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"
# O resto, sem reinstalar o torch (a linha do torch no requirements é para o computador local)
grep -vE "^(torch|torchvision)==" requirements.txt | pip install -q -r /dev/stdin
# Pesos da ResNet-50 baixados agora, no login: os nós de GPU podem não ter internet
python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V2)"
python -m ipykernel install --user --name python3 --display-name "Python 3 (macdeg)" >/dev/null
chmod 700 ~/macdeg_data 2>/dev/null || true
grep -q "miniforge3/etc/profile.d/conda.sh" ~/.bashrc || echo "source $BASE/etc/profile.d/conda.sh" >> ~/.bashrc
python -c "import torch, papermill, wandb; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| ok')"
