"""Remove as saídas de um notebook, para imagens e IDs não entrarem no git.

Só usa a biblioteca padrão, para funcionar com o Git do WSL e com o do Windows.
Lê de um arquivo (argumento) ou da entrada padrão e escreve na saída padrão.
Em caso de erro, termina com código diferente de zero e o Git recusa o arquivo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def clean(raw: bytes) -> bytes:
    notebook = json.loads(raw.decode("utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell["metadata"] = {}
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


if __name__ == "__main__":
    data = Path(sys.argv[1]).read_bytes() if len(sys.argv) > 1 else sys.stdin.buffer.read()
    sys.stdout.buffer.write(clean(data))
