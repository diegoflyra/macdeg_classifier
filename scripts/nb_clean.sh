#!/bin/sh
# Filtro do Git: acha um Python que funcione aqui (WSL ou Windows) e limpa o notebook.
# Sem Python, falha: com filter.required, o Git recusa o arquivo em vez de deixá-lo passar.
for candidate in .venv/bin/python python3 python py; do
  if "$candidate" -c "import json" >/dev/null 2>&1; then
    exec "$candidate" scripts/nb_clean.py "$@"
  fi
done
echo "nb_clean: nenhum Python encontrado; notebook recusado." >&2
exit 1
