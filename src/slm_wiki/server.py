from __future__ import annotations

import json
import mimetypes
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from slm_wiki.config import ROOT, Settings
from slm_wiki.ingest import ingest_path, pending_raw
from slm_wiki.lint import lint
from slm_wiki.llm import LLMClient, LLMError, fetch_mga_models
from slm_wiki.progress import Progress
from slm_wiki.query import query, save_answer, search
from slm_wiki.vault import WIKILINK_RE, Vault, norm, safe_title

STATIC = Path(__file__).parent / "static"
_ingesting = threading.Lock()  # one ingest at a time; queries may run alongside (llama.cpp has parallel slots)
_llama_starting = threading.Lock()
_mga_cache: tuple[float, list] = (0.0, [])  # ponytail: 10 min; the menu doesn't need a live catalog


def start_llama(settings: Settings) -> None:
    """Fire-and-forget: the script no-ops if already serving, so this is safe to call repeatedly."""
    if not _llama_starting.acquire(blocking=False):
        return
    try:
        log = (settings.logs_dir / "llama.out").open("a")
        subprocess.Popen(
            ["./scripts/start_llama.sh"], cwd=ROOT, stdout=log, stderr=log, start_new_session=True
        )
    finally:
        threading.Timer(5.0, _llama_starting.release).start()  # debounce repeat clicks, not a real lock


class Handler(BaseHTTPRequestHandler):
    s: Settings

    def log_message(self, *_: object) -> None:
        pass

    # ---- plumbing -------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: object, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _line(self, obj: object) -> None:
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _vault_file(self, rel: str) -> Path | None:
        """Resolve a vault-relative .md path, refusing anything outside the vault."""
        root = self.s.vault.resolve()
        p = (root / rel).resolve()
        return p if root in p.parents and p.suffix == ".md" and p.is_file() else None

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        route = {
            "/api/tree": self.tree,
            "/api/page": self.page,
            "/api/graph": self.graph,
            "/api/search": self.search,
            "/api/events": self.events,
            "/api/health": self.health,
        }.get(u.path)
        if route:
            route(q)
            return
        name = "index.html" if u.path in ("", "/") else u.path.lstrip("/")
        f = (STATIC / name).resolve()
        if STATIC.resolve() in f.parents and f.is_file():
            self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        route = {
            "/api/ask": self.ask,
            "/api/save": self.save,
            "/api/ingest": self.ingest,
            "/api/lint": self.lint,
            "/api/provider": self.provider,
        }
        fn = route.get(urlparse(self.path).path)
        if fn:
            fn(self._body())
        else:
            self._json({"error": "not found"}, 404)

    # ---- reads ----------------------------------------------------------
    def tree(self, _: dict) -> None:
        v = Vault(self.s)
        v.ensure_layout()
        pages = []
        for p in v.pages():
            meta = v.read(p)[0]
            pages.append({
                "path": v.rel(p), "title": p.stem, "kind": p.parent.name,
                "category": meta.get("category") or "", "summary": meta.get("summary") or "",
                "updated": meta.get("updated") or "", "answered": meta.get("answered") or "",
            })
        pending = set(pending_raw(v))
        raw = [{"path": v.rel(p), "title": p.name, "pending": p in pending} for p in v.raw_files()]
        self._json({"pages": pages, "raw": raw, "ingesting": _ingesting.locked()})

    def page(self, q: dict) -> None:
        v = Vault(self.s)
        path = self._vault_file(q.get("path", ""))
        if not path:
            self._json({"error": "no such page"}, 404)
            return
        meta, body = v.read(path)
        n = norm(path.stem)
        backlinks = [
            {"path": v.rel(p), "title": p.stem}
            for p in v.pages()
            if p != path and any(norm(m.group(1)) == n for m in WIKILINK_RE.finditer(p.read_text(encoding="utf-8")))
        ]
        self._json({"path": v.rel(path), "title": path.stem, "meta": meta, "body": body, "backlinks": backlinks})

    def graph(self, _: dict) -> None:
        v = Vault(self.s)
        pages = v.pages()
        by_norm = {norm(p.stem): v.rel(p) for p in pages}
        nodes, edges = [], set()
        for p in pages:
            meta, body = v.read(p)
            src = v.rel(p)
            nodes.append({"id": src, "title": p.stem, "kind": p.parent.name, "category": meta.get("category") or ""})
            for m in WIKILINK_RE.finditer(body):
                dst = by_norm.get(norm(m.group(1)))
                if dst and dst != src:
                    edges.add(tuple(sorted((src, dst))))
        self._json({"nodes": nodes, "edges": sorted(edges)})

    def search(self, q: dict) -> None:
        v = Vault(self.s)
        hits = search(v, q.get("q", ""), 20) if q.get("q") else []
        self._json([{"path": v.rel(p), "title": p.stem, "summary": v.read(p)[0].get("summary") or ""} for p in hits])

    def health(self, _: dict) -> None:
        global _mga_cache
        with LLMClient(self.s) as llm:
            info = llm.health()
        if info.get("mga_models") is not None:
            _mga_cache = (time.time(), info["mga_models"])
        elif not self.s.mga_enabled:
            _mga_cache = (0.0, [])
        elif not _mga_cache[1] or time.time() - _mga_cache[0] > 600:
            try:
                choices = fetch_mga_models(self.s)
            except Exception:  # noqa: BLE001 - keep the last good list if MGA is briefly unreachable
                choices = None
            if choices is not None:
                _mga_cache = (time.time(), choices)
        self._json({
            **info,
            "local_model": self.s.llm_model,
            "mga_models": _mga_cache[1],
            "ingesting": _ingesting.locked(),
            "provider": self.s.llm_provider,
            "mga_enabled": self.s.mga_enabled,
            "mga_configured": self.s.mga_enabled and bool(self.s.mga_api_key),
        })

    def provider(self, b: dict) -> None:
        p = str(b.get("provider") or "")
        if p not in ("local", "mga"):
            self._json({"error": "provider must be 'local' or 'mga'"}, 400)
            return
        if p == "mga" and not self.s.mga_enabled:
            self._json({"error": "MGA is disabled (SLM_WIKI_MGA_ENABLED=false in .env)"}, 403)
            return
        if p == "mga" and not self.s.mga_api_key:
            self._json({"error": "SLM_WIKI_MGA_API_KEY is not set in .env"}, 400)
            return
        self.s.llm_provider = p
        if p == "local":
            start_llama(self.s)
        elif model := str(b.get("model") or ""):
            self.s.mga_model = model
        self.health({})

    def events(self, _: dict) -> None:
        """SSE tail of logs/events.jsonl, so CLI runs show up live too."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        path = self.s.events_path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-40:] if path.exists() else []
            for ln in lines:
                self.wfile.write(f"data: {ln}\n\n".encode())
            pos = path.stat().st_size if path.exists() else 0
            idle = 0.0
            while True:
                time.sleep(0.5)
                size = path.stat().st_size if path.exists() else 0
                if size < pos:  # truncated/rotated
                    pos = 0
                if size > pos:
                    with path.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos)
                    pos = size
                    for ln in chunk.decode("utf-8", "replace").splitlines():
                        if ln.strip():
                            self.wfile.write(f"data: {ln}\n\n".encode())
                    idle = 0
                else:
                    idle += 0.5
                    if idle >= 15:
                        self.wfile.write(b": ping\n\n")
                        idle = 0
        except (BrokenPipeError, ConnectionResetError):
            return

    # ---- writes ---------------------------------------------------------
    def ask(self, b: dict) -> None:
        question = str(b.get("question") or "").strip()
        if not question:
            self._json({"error": "empty question"}, 400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            res = query(self.s, question, on_token=lambda t: self._line({"t": t}))
            self._line({"done": True, **res})
        except LLMError as e:
            self._line({"error": f"{e} — is llama-server running? (./scripts/start_llama.sh)"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def save(self, b: dict) -> None:
        question, answer = str(b.get("question") or "").strip(), str(b.get("answer") or "").strip()
        if not question or not answer:
            self._json({"error": "question and answer required"}, 400)
            return
        cited = [str(c) for c in b.get("cited") or []]
        self._json({"path": save_answer(self.s, question, answer, cited)})

    def ingest(self, b: dict) -> None:
        v = Vault(self.s)
        v.ensure_layout()
        force = bool(b.get("force"))
        if files := [f for f in b.get("files") or [] if isinstance(f, dict) and str(f.get("content") or "").strip()]:
            dests = [
                self.s.raw_dir / f"{safe_title(str(f.get('name') or 'note').removesuffix('.md').removesuffix('.txt'))}.md"
                for f in files
            ]
            if taken := [d.name for d in dests if d.exists()]:
                self._json({"error": f"already in raw/: {', '.join(taken)} (raw sources are immutable; rename)"}, 409)
                return
            for f, d in zip(files, dests, strict=True):
                d.write_text(str(f["content"]), encoding="utf-8")
            targets = dests
        elif b.get("path"):
            p = self._vault_file(str(b["path"]))
            if not p or self.s.raw_dir.resolve() not in p.parents:
                self._json({"error": "path must be a file in raw/"}, 400)
                return
            targets = [p]
        else:
            targets = v.raw_files() if force else pending_raw(v)
        if not targets:
            self._json({"started": [], "message": "nothing new to ingest"})
            return
        if not _ingesting.acquire(blocking=False):
            saved = " Your files are saved in raw/ as pending." if files else ""
            self._json({"error": f"An ingest is already running (see Activity).{saved}"}, 409)
            return

        def run() -> None:
            try:
                for t in targets:
                    ingest_path(self.s, t, force=force)
            except Exception as e:  # noqa: BLE001 - surface any failure in the Activity feed
                Progress(self.s, "ingest", quiet=True)("error", f"ingest crashed: {e}")
            finally:
                _ingesting.release()

        threading.Thread(target=run, daemon=True).start()
        self._json({"started": [v.rel(t) for t in targets]}, 202)

    def lint(self, _: dict) -> None:
        self._json(lint(self.s))


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    Handler.s = settings
    Vault(settings).ensure_layout()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
