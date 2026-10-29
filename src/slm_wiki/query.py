from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from slm_wiki.config import Settings
from slm_wiki.llm import LLMClient
from slm_wiki.progress import Progress
from slm_wiki.vault import WIKILINK_RE, Vault, norm, render_page, safe_title

ANSWER_SYS = """You are the librarian of a personal wiki. Answer ONLY from the WIKI PAGES given.

Write markdown with exactly these sections:
## Answer
Direct answer first, then supporting detail. Every factual sentence ends with a citation like [[Page Title]].
## Evidence
- short quote or paraphrase — [[Page Title]]
## Gaps
- what the pages do not cover (or "None")

Cite only titles that appear as page headers below. Never invent pages or facts."""

STOP = {"the", "and", "for", "are", "was", "were", "what", "when", "where", "which", "who", "whom", "how",
        "did", "does", "about", "with", "from", "that", "this", "have", "has", "into", "your", "you", "our"}


def search(vault: Vault, question: str, k: int) -> list[Path]:
    # ponytail: FTS index rebuilt per query — fine to a few thousand pages; persist it if that gets slow.
    pages = vault.pages()
    terms = [t for t in re.findall(r"\w{3,}", question.lower()) if t not in STOP]
    if not pages or not terms:
        return pages[:k]
    db = sqlite3.connect(":memory:")
    db.execute("create virtual table p using fts5(title, body, idx unindexed)")
    db.executemany(
        "insert into p values (?, ?, ?)",
        [(p.stem, p.read_text(encoding="utf-8"), i) for i, p in enumerate(pages)],
    )
    match = " OR ".join(f'"{t}"' for t in terms)
    rows = db.execute(
        "select idx from p where p match ? order by bm25(p, 10.0, 1.0) limit ?", (match, k)
    ).fetchall()
    return [pages[r[0]] for r in rows]


def build_context(vault: Vault, hits: list[Path], budget: int) -> tuple[str, list[Path], list[Path]]:
    parts, used, clipped, total = [], [], [], 0
    for p in hits:
        # ponytail: head-truncate big pages to a third of the budget; chunk-level retrieval if tails matter.
        body = vault.read(p)[1]
        room = min(budget - total, budget // 3)
        if room < 800:
            clipped.append(p)
            continue
        if len(body) > room:
            body = body[:room] + "\n[clipped]"
            clipped.append(p)
        parts.append(f"=== PAGE: {p.stem} ({vault.rel(p)}) ===\n{body}")
        used.append(p)
        total += len(body)
    return "\n\n".join(parts), used, clipped


def check_citations(answer: str, used: list[Path]) -> tuple[list[str], list[str]]:
    by_norm = {norm(p.stem): p.stem for p in used}
    valid, invalid = [], []
    for m in WIKILINK_RE.finditer(answer):
        t = m.group(1).strip()
        stem = by_norm.get(norm(t))
        bucket, val = (valid, stem) if stem else (invalid, t)
        if val not in bucket:
            bucket.append(val)
    return valid, invalid


def answer_sections(question: str, answer: str) -> list[dict]:
    """Question + the answer's own `## ` sections, so headings aren't nested under a wrapper."""
    parts = [p.partition("\n") for p in re.split(r"(?m)^## ", answer) if p.strip()]
    return [{"h2": "Question", "body": question}] + [{"h2": h.strip(), "body": b} for h, _, b in parts]


def query(
    settings: Settings,
    question: str,
    *,
    on_token: Callable[[str], None] = lambda _: None,
    file_answer: bool = False,
) -> dict:
    vault = Vault(settings)
    vault.ensure_layout()
    log = Progress(settings, "query")
    glossary = vault.glossary()
    expand = [t for t, _, al in glossary if any(re.search(rf"\b{re.escape(a)}\b", question, re.IGNORECASE) for a in [t, *al])]
    hits = search(vault, " ".join([question, *expand]), settings.query_top_k)
    context, used, clipped = build_context(vault, hits, settings.query_char_budget)
    log("context", f"{len(used)} pages, {len(context)} chars"
        + (f", clipped: {', '.join(p.stem for p in clipped)}" if clipped else ""))

    parts: list[str] = []
    with LLMClient(settings) as llm:
        gloss = "\n".join(f"- {t}: {m}" for t, m, _ in glossary if t in expand)
        prompt = f"QUESTION: {question}\n\n" + (f"GLOSSARY:\n{gloss}\n\n" if gloss else "")
        for delta in llm.chat_stream(ANSWER_SYS, f"{prompt}WIKI PAGES:\n\n{context or '(empty)'}"):
            parts.append(delta)
            on_token(delta)
    answer = "".join(parts).strip()
    valid, invalid = check_citations(answer, used)
    if invalid:
        log("citations", f"unknown citations: {', '.join(invalid)}")

    vault.append_log("query", question[:100], f"read: {', '.join(f'[[{p.stem}]]' for p in used)}")
    log("done", f"answered from {len(used)} pages, {len(valid)} cited")
    return {
        "answer": answer,
        "cited": valid,
        "unknown_citations": invalid,
        "pages_read": [vault.rel(p) for p in used],
        "clipped": [vault.rel(p) for p in clipped],
        "filed": save_answer(settings, question, answer, valid) if file_answer and answer else None,
    }


def save_answer(settings: Settings, question: str, answer: str, cited: list[str]) -> str:
    """File an answer as a wiki/queries page so explorations compound; returns its vault path."""
    vault = Vault(settings)
    path = settings.wiki_dir / "queries" / f"{safe_title(question)[:80]}.md"
    path.write_text(
        render_page(
            kind="queries",
            title=path.stem,
            summary=question,
            sections=answer_sections(question, answer),
            sources=cited,
        ),
        encoding="utf-8",
    )
    vault.relink([path])
    vault.rebuild_index()
    vault.append_log("save", path.stem, f"[[{path.stem}]]")
    return vault.rel(path)
