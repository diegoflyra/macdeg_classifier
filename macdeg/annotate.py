"""Programa local para marcar o centro do disco óptico com um clique.

Uso: python -m macdeg.annotate [--cache cache] [--out annotations/disc_clicks.csv] [--port 8765]
Depois, abra http://localhost:8765 no navegador.

Os dados não saem da máquina: o servidor só escuta em 127.0.0.1 e a página não carrega nada de fora.
Para não influenciar a marcação, a página não mostra classe, nome de arquivo nem o palpite do detector.
A ordem é sorteada, e parte das imagens reaparece sem aviso para medir a repetibilidade dos cliques.

O CSV só recebe linhas novas. Se a mesma imagem for marcada de novo, vale a linha mais recente.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from macdeg.localization import DD_PRIOR_R

FIELDS = ["file", "repeat", "status", "x_native", "y_native", "seconds", "timestamp"]
STATUSES = ("clicked", "not_visible")
DISPLAY_MAX_SIDE = 1600
REPEAT_FRAC = 0.03
REPEAT_MIN_GAP = 100
IMAGE_CACHE_SIZE = 6


@dataclass(frozen=True)
class Item:
    file: str
    repeat: int  # 0 na primeira vez, 1 na repetição sem aviso
    path: str
    fov_radius: float


def build_queue(metadata: pd.DataFrame, seed: int = 0, repeat_frac: float = REPEAT_FRAC,
                min_gap: int = REPEAT_MIN_GAP) -> list[Item]:
    """Fila sorteada e reproduzível. Qualquer trecho inicial é uma amostra aleatória da base."""
    files = metadata.drop_duplicates("file").sort_values("file")
    rng = np.random.default_rng(seed)
    rows = [files.iloc[i] for i in rng.permutation(len(files))]
    queue = [Item(str(r["file"]), 0, str(r["original_path"]), float(r["fov_radius"])) for r in rows]
    if not queue:
        return queue

    # Só repete imagens com espaço depois delas, para a repetição não vir logo em seguida.
    eligible = max(len(queue) - min_gap, 0) or len(queue)
    n_repeats = min(int(round(len(queue) * repeat_frac)), eligible)
    for index in sorted(rng.choice(eligible, n_repeats, replace=False).tolist(), reverse=True):
        first = queue[index]
        low = index + min_gap
        position = int(rng.integers(low, len(queue) + 1)) if low < len(queue) else len(queue)
        queue.insert(position, Item(first.file, 1, first.path, first.fov_radius))
    return queue


def load_answers(path: Path) -> dict[tuple[str, int], dict]:
    """Última resposta de cada (arquivo, repetição). Ignora uma linha cortada por queda do programa."""
    answers: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return answers
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("timestamp") and row.get("status") in STATUSES:
                answers[(row["file"], int(row["repeat"]))] = row
    return answers


def render_jpeg(path: Path, max_side: int = DISPLAY_MAX_SIDE) -> tuple[bytes, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError("imagem ilegível")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("falha ao codificar a imagem")
    return encoded.tobytes(), width, height


class Session:
    def __init__(self, queue: list[Item], out: Path, root: Path):
        self.queue, self.out, self.root = queue, out, root
        self.answers = load_answers(out)
        self.lock = threading.Lock()
        self.images: OrderedDict[str, tuple[bytes, int, int]] = OrderedDict()
        out.parent.mkdir(parents=True, exist_ok=True)
        new_file = not out.exists() or out.stat().st_size == 0
        if not new_file and not out.read_bytes().endswith(b"\n"):
            with out.open("a", encoding="utf-8") as handle:
                handle.write("\n")
        if new_file:
            with out.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(FIELDS)

    def image(self, position: int) -> tuple[bytes, int, int]:
        item = self.queue[position]
        with self.lock:
            if item.file in self.images:
                self.images.move_to_end(item.file)
                return self.images[item.file]
        rendered = render_jpeg(self.root / item.path)
        with self.lock:
            self.images[item.file] = rendered
            while len(self.images) > IMAGE_CACHE_SIZE:
                self.images.popitem(last=False)
        return rendered

    def state(self) -> dict:
        with self.lock:
            done = sum((i.file, i.repeat) in self.answers for i in self.queue)
            pending = next((p for p, i in enumerate(self.queue) if (i.file, i.repeat) not in self.answers), None)
            times = [float(a["seconds"]) for a in self.answers.values() if a.get("seconds")]
        return {"total": len(self.queue), "done": done, "next": pending,
                "median_seconds": float(np.median(times)) if times else None}

    def item(self, position: int) -> dict:
        """Só o necessário para desenhar a página: nunca o nome do arquivo."""
        item = self.queue[position]
        _, width, height = self.image(position)
        with self.lock:
            answer = self.answers.get((item.file, item.repeat))
        previous = None
        if answer:
            previous = {"status": answer["status"]}
            if answer["status"] == "clicked":
                previous.update(fx=float(answer["x_native"]) / width, fy=float(answer["y_native"]) / height)
        return {"position": position, "ring_fx": DD_PRIOR_R * item.fov_radius / width, "previous": previous}

    def record(self, position: int, status: str, fx: float | None, fy: float | None, seconds: float) -> None:
        if status not in STATUSES:
            raise ValueError("status inválido")
        item = self.queue[position]
        _, width, height = self.image(position)
        x, y = "", ""
        if status == "clicked":
            if fx is None or fy is None or not (0 <= fx <= 1 and 0 <= fy <= 1):
                raise ValueError("clique fora da imagem")
            x, y = round(fx * width, 1), round(fy * height, 1)
        row = {"file": item.file, "repeat": item.repeat, "status": status, "x_native": x, "y_native": y,
               "seconds": round(max(seconds, 0.0), 2), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with self.lock:
            with self.out.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, FIELDS).writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            self.answers[(item.file, item.repeat)] = {k: str(v) for k, v in row.items()}


PAGE = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Marcação do disco óptico</title>
<style>
html, body { margin: 0; height: 100%; background: #000; color: #ddd; font: 14px system-ui, sans-serif; overflow: hidden; }
#bar { height: 40px; display: flex; align-items: center; gap: 18px; padding: 0 14px; background: #1b1b1b; }
#bar button { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; cursor: pointer; font: inherit; }
#bar .grow { flex: 1; }
#stage { position: relative; height: calc(100% - 40px); display: flex; align-items: center; justify-content: center; }
#wrap { position: relative; line-height: 0; }
#photo { max-width: 100vw; max-height: calc(100vh - 40px); cursor: none; user-select: none; }
#ring, #mark { position: absolute; border-radius: 50%; pointer-events: none; transform: translate(-50%, -50%); display: none; }
#ring { border: 1px solid #0ff; box-shadow: 0 0 0 1px #0008; }
#ring::after { content: ""; position: absolute; left: 50%; top: 50%; width: 4px; height: 4px; margin: -2px; border-radius: 50%; background: #0ff; }
#mark { width: 12px; height: 12px; background: #3f6; border: 1px solid #000; }
#note { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); background: #000c; padding: 6px 12px; line-height: 1.3; display: none; }
#finished { display: none; padding: 40px; font-size: 18px; }
</style></head><body>
<div id="bar">
  <span id="progress"></span><span id="eta"></span><span class="grow"></span>
  <span>Clique no centro do disco</span>
  <button id="none">N: não consigo ver</button>
  <button id="back">Z: voltar</button>
</div>
<div id="stage">
  <div id="wrap"><img id="photo" alt="" draggable="false"><div id="ring"></div><div id="mark"></div></div>
  <div id="note"></div>
  <div id="finished">Marcação concluída. Pode fechar esta janela.</div>
</div>
<script>
const photo = document.getElementById('photo'), ring = document.getElementById('ring'), mark = document.getElementById('mark');
const note = document.getElementById('note');
let position = 0, item = null, ready = false, shownAt = 0, total = 0;
const GUARD_MS = 250;  // evita que um clique duplo marque também a imagem seguinte

async function refreshState() {
  const s = await (await fetch('/api/state')).json();
  total = s.total;
  document.getElementById('progress').textContent = s.done + ' de ' + s.total;
  if (s.median_seconds) {
    const minutes = Math.round((s.total - s.done) * s.median_seconds / 60);
    document.getElementById('eta').textContent = s.median_seconds.toFixed(1) + ' s por imagem; faltam cerca de ' + minutes + ' min';
  }
  return s;
}
function finish() {
  document.getElementById('wrap').style.display = 'none';
  document.getElementById('finished').style.display = 'block';
}
async function show(next) {
  ready = false;
  mark.style.display = 'none'; ring.style.display = 'none'; note.style.display = 'none';
  position = next;
  document.getElementById('wrap').style.display = '';
  document.getElementById('finished').style.display = 'none';
  item = await (await fetch('/api/item/' + position)).json();
  photo.onload = () => {
    ready = true; shownAt = performance.now();
    ring.style.width = ring.style.height = (item.ring_fx * photo.clientWidth) + 'px';
    const p = item.previous;
    if (p && p.status === 'clicked') {
      mark.style.left = (p.fx * photo.clientWidth) + 'px';
      mark.style.top = (p.fy * photo.clientHeight) + 'px';
      mark.style.display = 'block';
    }
    if (p) { note.textContent = p.status === 'clicked' ? 'Já marcada. Clique de novo para corrigir.' : 'Marcada como não visível. Clique para corrigir.'; note.style.display = 'block'; }
    [1, 2].forEach(k => { if (position + k < total) new Image().src = '/image/' + (position + k); });
  };
  photo.src = '/image/' + position;
}
async function answer(status, fx, fy) {
  if (!ready || performance.now() - shownAt < GUARD_MS) return;
  ready = false;
  const seconds = (performance.now() - shownAt) / 1000;
  const r = await fetch('/api/answer', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position, status, fx, fy, seconds }) });
  if (!r.ok) { note.textContent = 'Erro ao salvar. Nada foi gravado; tente de novo.'; note.style.display = 'block'; ready = true; return; }
  const s = await refreshState();
  if (position + 1 < total) show(position + 1);
  else if (s.next !== null) show(s.next);
  else finish();
}
photo.addEventListener('click', e => {
  const b = photo.getBoundingClientRect();
  answer('clicked', (e.clientX - b.left) / b.width, (e.clientY - b.top) / b.height);
});
photo.addEventListener('mousemove', e => {
  const b = photo.getBoundingClientRect();
  ring.style.left = (e.clientX - b.left) + 'px'; ring.style.top = (e.clientY - b.top) + 'px';
  ring.style.display = 'block';
});
photo.addEventListener('mouseleave', () => { ring.style.display = 'none'; });
function goBack() { if (position > 0) show(position - 1); }
document.getElementById('none').addEventListener('click', () => answer('not_visible', null, null));
document.getElementById('back').addEventListener('click', goBack);
document.addEventListener('keydown', e => {
  const key = e.key.toLowerCase();
  if (key === 'n') answer('not_visible', null, null);
  if (key === 'z' || key === 'backspace') { e.preventDefault(); goBack(); }
});
window.addEventListener('resize', () => { if (item) show(position); });
refreshState().then(s => { if (s.next === null) { if (s.total) finish(); } else show(s.next); });
</script></body></html>
"""


def make_handler(session: Session, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # sem log de acesso
            pass

        def _send(self, status: int, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _host_ok(self) -> bool:
            # Impede que uma página externa alcance o servidor por um domínio apontado para 127.0.0.1.
            if self.headers.get("Host") in allowed_hosts:
                return True
            self._json({"error": "host não permitido"}, 403)
            return False

        def _position(self, text: str) -> int | None:
            if text.isdigit() and int(text) < len(session.queue):
                return int(text)
            self._json({"error": "posição inválida"}, 404)
            return None

        def do_GET(self):
            if not self._host_ok():
                return
            try:
                if self.path == "/":
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self._json(session.state())
                elif self.path.startswith("/api/item/"):
                    position = self._position(self.path.rsplit("/", 1)[1])
                    if position is not None:
                        self._json(session.item(position))
                elif self.path.startswith("/image/"):
                    position = self._position(self.path.rsplit("/", 1)[1])
                    if position is not None:
                        self._send(200, session.image(position)[0], "image/jpeg", cache=True)
                else:
                    self._json({"error": "não encontrado"}, 404)
            except OSError:
                self._json({"error": "falha ao ler a imagem"}, 500)

        def do_POST(self):
            if not self._host_ok():
                return
            # Formulários de outras páginas não conseguem enviar este tipo de conteúdo sem autorização.
            if self.path != "/api/answer" or self.headers.get("Content-Type") != "application/json":
                self._json({"error": "requisição inválida"}, 400)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                position = self._position(str(body["position"]))
                if position is None:
                    return
                session.record(position, body["status"], body.get("fx"), body.get("fy"), float(body.get("seconds", 0)))
                self._json({"ok": True})
            except (KeyError, TypeError, ValueError):
                self._json({"error": "resposta inválida"}, 400)
            except OSError:
                self._json({"error": "falha ao gravar"}, 500)

    return Handler


def make_server(session: Session, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), None)
    server.RequestHandlerClass = make_handler(session, server.server_address[1])
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=Path("annotations/disc_clicks.csv"))
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    metadata = pd.read_csv(args.cache / "metadata.csv", usecols=["file", "original_path", "fov_radius"])
    session = Session(build_queue(metadata), args.out, args.root)
    state = session.state()
    server = make_server(session, args.port)
    print(f"Marcadas: {state['done']} de {state['total']}. Respostas em {args.out}.")
    print(f"Abra http://localhost:{server.server_address[1]} no navegador. Ctrl+C encerra; tudo já fica salvo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
