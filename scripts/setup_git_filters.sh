#!/bin/sh
# Configura este clone para remover as saídas dos notebooks. Rode uma vez, na raiz do projeto.
# Serve para o Git do WSL e para o do Windows, pois não grava caminhos absolutos.
set -e
git config --local --remove-section filter.nbstripout 2>/dev/null || true
git config --local --remove-section diff.ipynb 2>/dev/null || true
git config --local filter.notebook-clean.clean "sh scripts/nb_clean.sh"
git config --local filter.notebook-clean.smudge cat
git config --local filter.notebook-clean.required true
git config --local diff.notebook-clean.textconv "sh scripts/nb_clean.sh"
attributes="$(git rev-parse --git-path info/attributes)"
mkdir -p "$(dirname "$attributes")"
printf '*.ipynb filter=notebook-clean diff=notebook-clean\n' > "$attributes"
echo "Filtro de notebooks configurado."
