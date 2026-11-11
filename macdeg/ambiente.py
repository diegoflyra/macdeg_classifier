"""Segredos e envio de resultados ao GitHub, no formato do mlp_fase5.

Segredos vêm de variáveis de ambiente ou de um .env (GITHUB_TOKEN, WANDB_API_KEY, HF_TOKEN,
WANDB_ENTITY). Nunca são impressos nem gravados em arquivo de resultado.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS_BRANCH = "resultados"


def load_dotenv(path: Path | str | None = None) -> None:
    """Lê CHAVE=valor; nunca sobrescreve variáveis já definidas. Padrão: $MACDEG_ENV (o lançador aponta
    para o .env do clone no $HOME) ou o .env da raiz do projeto."""
    path = Path(path or os.environ.get("MACDEG_ENV") or PROJECT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#") and value.strip() and not os.environ.get(key):
            os.environ[key] = value.strip().strip("'").strip('"')


def secret(name: str) -> str:
    return os.environ.get(name, "").strip()


def which_secrets() -> dict:
    """Só diz quais segredos existem, sem mostrar o valor."""
    return {name: bool(secret(name)) for name in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN", "WANDB_ENTITY")}


def _authed(url: str, token: str) -> str:
    return url.replace("https://", f"https://x-access-token:{token}@", 1) if token else url


def _git(*args, cwd=None, check=True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        # A mensagem do git pode conter a URL com o token: não repassar o texto.
        raise RuntimeError(f"git {args[0]} falhou (código {result.returncode})")
    return result


def repo_url() -> str:
    return os.environ.get("MACDEG_REPO_URL") or _git("remote", "get-url", "origin", cwd=PROJECT).stdout.strip()


def push_results(local_dir: Path | str, target: str, message: str, url: str | None = None,
                 token: str | None = None, branch: str = RESULTS_BRANCH) -> bool:
    """Copia `local_dir` para `target` (ex.: resultados/referencia-20261112-1030) na branch órfã de
    resultados e faz push. Cria a branch se ela não existir. Devolve False se falhar."""
    token = token if token is not None else secret("GITHUB_TOKEN")
    url = url or repo_url()
    if not token:
        print("GitHub: sem GITHUB_TOKEN, nada enviado.")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        authed = _authed(url, token)
        cloned = _git("clone", "-q", "--depth", "1", "--branch", branch, authed, str(work), check=False)
        if cloned.returncode != 0:
            work.mkdir()
            _git("init", "-q", cwd=work)
            _git("checkout", "-q", "--orphan", branch, cwd=work)
            _git("remote", "add", "origin", authed, cwd=work)
        _git("config", "user.name", "macdeg-cluster", cwd=work)
        _git("config", "user.email", "macdeg-cluster@users.noreply.github.com", cwd=work)
        _git("config", "commit.gpgsign", "false", cwd=work)  # commit automático: sem assinatura (pediria senha)
        destination = work / target
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(local_dir, destination)
        _git("add", "-A", cwd=work)
        if not _git("status", "--porcelain", cwd=work).stdout.strip():
            return True
        _git("commit", "-q", "-m", message, cwd=work)
        for _ in range(3):  # outra tarefa pode ter enviado antes: rebase e tenta de novo
            if _git("push", "-q", "origin", f"HEAD:{branch}", cwd=work, check=False).returncode == 0:
                return True
            _git("pull", "-q", "--rebase", "origin", branch, cwd=work, check=False)
        print("GitHub: o push falhou depois de 3 tentativas.")
        return False


def fetch_results(target: str, destination: Path | str, url: str | None = None, token: str | None = None,
                  branch: str = RESULTS_BRANCH) -> Path:
    """Baixa `target` da branch de resultados (para continuar de uma fase anterior ou comparar)."""
    token = token if token is not None else secret("GITHUB_TOKEN")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        _git("clone", "-q", "--depth", "1", "--branch", branch, _authed(url or repo_url(), token), str(work))
        source = work / target
        if not source.exists():
            raise FileNotFoundError(f"{target} não existe na branch {branch}")
        destination = Path(destination)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def commit_info() -> dict:
    head = _git("rev-parse", "HEAD", cwd=PROJECT, check=False).stdout.strip()
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=PROJECT, check=False).stdout.strip())
    return {"commit": head, "codigo_alterado_sem_commit": dirty}
