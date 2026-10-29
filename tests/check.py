"""Offline self-check (no LLM): uv run python tests/check.py"""

import json
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from slm_wiki import ingest as ingest_mod
from slm_wiki.config import Settings
from slm_wiki.ingest import chunk_markdown, pending_raw, raw_sha
from slm_wiki.lint import lint
from slm_wiki.llm import LLMClient, fetch_mga_models, latest_chat_models
from slm_wiki.query import answer_sections, check_citations, search
from slm_wiki.server import serve
from slm_wiki.vault import Vault, parse_frontmatter, render_page, safe_title

assert safe_title('A/B: "x"?') == "A B x"

doc = "# A\n" + "a" * 50 + "\n# B\n" + "b" * 50 + "\n# C\n" + "c" * 300
chunks = chunk_markdown(doc, 120)
assert chunks[0].startswith("# A") and all(len(c) <= 120 for c in chunks), chunks
assert "".join(chunks) == doc

md = render_page(kind="entities", title="CodeQL", summary='says "hi"', sections=[{"h2": "Notes", "body": "uses [[SAST tools]]"}],
                 sources=["Report", "Report"], disagreements=["x vs y"])
meta, body = parse_frontmatter(md)
assert meta["summary"] == 'says "hi"' and meta["sources"] == ["[[Report]]"], meta
assert "## Disagreements" in body and body.count("[[Report]]") == 1

with tempfile.TemporaryDirectory() as d:
    v = Vault(Settings(vault=Path(d)))
    v.ensure_layout()
    real = v.s.wiki_dir / "entities" / "Real.md"
    real.write_text("---\ntype: \"entity\"\n---\n# Real\n\nSees [[Ghost]] and [[Real|it]] again.\n", encoding="utf-8")
    v.delink_broken([real])
    assert "[[Ghost]]" not in real.read_text() and "Ghost" in real.read_text(), real.read_text()
    assert "[[Real|it]]" in real.read_text(), real.read_text()  # link to an existing page survives

with tempfile.TemporaryDirectory() as d:
    s = Settings(vault=Path(d))
    v = Vault(s)
    v.ensure_layout()
    (s.wiki_dir / "entities" / "SAST Tools.md").write_text(render_page(kind="entities", title="SAST Tools", summary="static analysis", sections=[]))
    page = s.wiki_dir / "entities" / "CodeQL.md"
    page.write_text(md)
    v.relink([page])
    assert "[[SAST Tools|SAST tools]]" in page.read_text(), page.read_text()

    hits = search(v, "what is static analysis?", 3)
    assert hits and hits[0].stem == "SAST Tools", hits
    assert check_citations("x [[sast tools]] y [[Nope]]", hits) == (["SAST Tools"], ["Nope"])

    report = lint(s)
    assert "[[CodeQL]] → Report" in report["broken_links"], report
    assert report["disagreements"] == ["CodeQL"] and report["orphans"] == ["CodeQL"], report
    v.rebuild_index()
    assert "- [[CodeQL]] — says" in s.index_path.read_text()

    s.schema_dir.mkdir()
    (s.schema_dir / "glossary.md").write_text(
        "| Term | Meaning | Aliases |\n|---|---|---|\n| SAST Tools | static analysis | SAST, SLM-based SAST |\n")
    g = v.glossary()
    assert g == [("SAST Tools", "static analysis", ["SAST", "SLM-based SAST"])], g
    assert v.canonical("slm based sast", g) == "SAST Tools" and v.canonical("Qwen", g) == "Qwen"
    page.write_text("see [[SAST]]")
    v.relink([page])
    assert page.read_text() == "see [[SAST Tools|SAST]]", page.read_text()

    raw = s.raw_dir / "note.md"
    raw.write_text("hello")
    assert pending_raw(v) == [raw]
    (s.wiki_dir / "sources" / "Note.md").write_text(render_page(
        kind="sources", title="Note", summary="", sections=[], extra={"raw": "raw/note.md", "raw_sha": raw_sha("hello")}))
    assert pending_raw(v) == []

    httpd = serve(s, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    tree = json.load(urlopen(f"{base}/api/tree"))
    assert {"raw/note.md"} == {r["path"] for r in tree["raw"]} and not tree["raw"][0]["pending"], tree
    assert json.load(urlopen(f"{base}/api/page?path=wiki/entities/CodeQL.md"))["backlinks"] == []
    for bad in ("../../etc/passwd", "/etc/hosts", "wiki/../../x.md"):
        try:
            urlopen(f"{base}/api/page?path={bad}")
            raise AssertionError(f"served {bad}")
        except HTTPError as e:
            assert e.code == 404, bad
    httpd.shutdown()

secs = answer_sections("Q?", "## Answer\nYes [[A]].\n## Gaps\n- None\n")
assert [x["h2"] for x in secs] == ["Question", "Answer", "Gaps"] and secs[1]["body"] == "Yes [[A]].\n", secs

ids = [x["id"] for x in latest_chat_models([
    {"id": "claude-opus-4.5", "name": "Opus 4.5", "model_status": "available", "model_type": "chat_completion"},
    {"id": "claude-opus-5.5-bedrock", "name": "Opus 5.5 Bedrock", "model_status": "available", "model_type": "chat_completion"},
    {"id": "claude-opus-5.5", "name": "Opus 5.5", "model_status": "available", "model_type": "chat_completion"},
    {"id": "gpt-4", "name": "old", "model_status": "deprecated", "model_type": "chat_completion"},
    {"id": "gpt-4o", "name": "alias", "model_status": "available", "model_type": "chat_completion", "is_alias": True},
    {"id": "text-embedding-3-small", "name": "emb", "model_status": "available", "model_type": "embedding"},
    {"id": "gemini-3-pro-image", "name": "img", "model_status": "available", "model_type": "chat_completion"},
])]
assert ids == ["claude-opus-5.5"], ids

m = Settings(_env_file=None, llm_provider="mga", mga_api_key="k", mga_model="m")
assert (m.effective_model, m.effective_api_key, m.effective_base_url) == ("m", "k", m.mga_base_url), m
assert m.llm_model != "m", "switching provider must not mutate the local triple"
assert Settings(_env_file=None).mga_enabled, "MGA must default to enabled"
assert fetch_mga_models(Settings(_env_file=None, mga_enabled=False, mga_api_key="k")) == []
try:
    Settings(_env_file=None, llm_provider="mga", mga_api_key="k", mga_enabled=False)
    raise AssertionError("provider=mga with MGA disabled must fail")
except ValueError:
    pass

replies, budgets = iter(['{"a": "bad "quote"}', '{"a": 1}']), []
fake = type("Fake", (LLMClient,), {"chat": lambda self, s, u, **kw: budgets.append(kw["max_tokens"]) or next(replies)})
assert fake.chat_json(object.__new__(fake), "s", "u", max_tokens=100) == {"a": 1} and budgets == [100, 200], budgets

extracted, updated = [], []
class FakeLLM:
    def __init__(self, _s): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def chat_json(self, system, user, **_):
        if system == ingest_mod.EXTRACT_SYS:
            week = user.rsplit("## ", 1)[1].split("\n")[0]
            extracted.append(week)
            return {"claims": [f"did {week}"], "entities": [{"name": week, "facts": [f"{week} fact"]}]}
        if system == ingest_mod.UPDATE_SYS:
            updated.append(user.split("PAGE: ")[1].split("\n")[0])
        return {"summary": "s"}
ingest_mod.LLMClient = FakeLLM
with tempfile.TemporaryDirectory() as d:
    s = Settings(_env_file=None, vault=Path(d), chunk_chars=60)
    raw = s.raw_dir / "log.md"
    s.raw_dir.mkdir(parents=True)
    raw.write_text("# Log\n## W1\n" + "a" * 30 + "\n## W2\n" + "b" * 30 + "\n")
    ingest_mod.ingest_path(s, raw)
    assert extracted == ["W1", "W2"] and sorted(updated) == ["W1", "W2"], (extracted, updated)
    extracted.clear(); updated.clear()
    raw.write_text(raw.read_text() + "## W3\n" + "c" * 30 + "\n")
    ingest_mod.ingest_path(s, raw)
    assert extracted == ["W3"] and updated == ["W3"], (extracted, updated)
    src = (s.wiki_dir / "sources" / "Log.md").read_text()
    assert all(f"[[{w}]]" in src and f"did {w}" in src for w in ("W1", "W2", "W3")), src

print("ok")
