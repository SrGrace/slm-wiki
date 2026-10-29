from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

import typer

from slm_wiki.config import get_settings
from slm_wiki.ingest import ingest_path, ingest_pending
from slm_wiki.lint import lint as run_lint
from slm_wiki.llm import LLMClient
from slm_wiki.query import query as run_query
from slm_wiki.server import start_llama
from slm_wiki.server import serve as make_server
from slm_wiki.vault import Vault

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Karpathy-style LLM wiki on local SLMs.")


@app.command()
def serve(
    port: int = typer.Option(8765, help="Port for the web UI."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser tab."),
) -> None:
    """Web UI: browse, ask, ingest, lint, live activity."""
    settings = get_settings()
    httpd = make_server(settings, port=port)
    if settings.llm_provider == "local":
        start_llama(settings)
    url = f"http://127.0.0.1:{port}"
    typer.echo(f"SLM Wiki on {url}  (Ctrl+C to stop)", err=True)
    if not no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


@app.command()
def health() -> None:
    """Check the llama.cpp endpoint."""
    with LLMClient(get_settings()) as llm:
        info = llm.health()
    typer.echo(json.dumps(info, indent=2))
    raise typer.Exit(0 if info["ok"] else 1)


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="e.g. raw/note.md (vault-relative). Omit to ingest all new/changed raw files."),  # noqa: B008
    force: bool = typer.Option(False, help="Re-ingest even if unchanged."),
) -> None:
    """Compile raw sources into wiki pages. Progress → stderr + vault/logs/events.jsonl."""
    s = get_settings()
    results = [ingest_path(s, path, force=force)] if path else ingest_pending(s, force=force)
    typer.echo(json.dumps(results, indent=2))


@app.command()
def ask(
    question: str,
    file: bool = typer.Option(False, "--file", help="Save the answer as a wiki/queries/ page."),
) -> None:
    """Ask the wiki; answer streams with [[citations]]."""
    s = get_settings()
    res = run_query(s, question, on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()), file_answer=file)
    typer.echo("\n")
    typer.echo(f"read: {', '.join(res['pages_read']) or 'nothing'}", err=True)
    if res["unknown_citations"]:
        typer.echo(f"⚠ cited pages not in context: {', '.join(res['unknown_citations'])}", err=True)
    if res["filed"]:
        typer.echo(f"filed: {res['filed']}", err=True)


@app.command()
def lint() -> None:
    """Deterministic health check → wiki/Lint Report.md."""
    s = get_settings()
    report = run_lint(s)
    typer.echo(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=2))
    typer.echo("details: wiki/Lint Report.md", err=True)


@app.command()
def reindex() -> None:
    """Rebuild wiki/index.md from page frontmatter."""
    Vault(get_settings()).rebuild_index()


if __name__ == "__main__":
    app()
