from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from slm_wiki.config import Settings
from slm_wiki.llm import LLMClient, LLMError
from slm_wiki.progress import Progress
from slm_wiki.vault import (
    WIKILINK_RE,
    Vault,
    norm,
    parse_frontmatter,
    render_page,
    section_bullets,
)

EXTRACT_SYS = """You extract knowledge from part of a source document for a personal wiki.
Return JSON:
{
  "claims": ["atomic, self-contained fact; keep dates, numbers, names"],
  "entities": [{"name": "...", "category": "person|project|tool|model|org|other", "summary": "one line", "facts": ["..."]}],
  "concepts": [{"name": "idea/method/theme", "summary": "one line", "facts": ["..."]}]
}
Rules:
- Only facts stated in the text; no speculation. Skip trivial mentions.
- Every claim and fact must stand alone: start it with the date or week of its section heading
  when the heading has one, e.g. "Week 6 (8 Jun - 12 Jun 2026): ...". Keep status words (done, TODO, blocked).
- Use the GLOSSARY: expand acronyms exactly as defined there and name items by their glossary Term.
- If an item matches a KNOWN PAGE, use that exact name. At most 8 entities and 6 concepts."""

SUMMARY_SYS = """Given claims extracted from one source, return JSON:
{"title": "short descriptive title", "summary": "3-5 sentences: what the source is, period covered, main threads",
 "contradictions": ["two claims that cannot both be true at the same time"]}
A status changing over time (TODO -> done, planned -> finished) is NOT a contradiction. Usually return []."""

UPDATE_SYS = """You maintain ONE wiki page. Merge NEW FACTS into the EXISTING PAGE.
Return JSON:
{
  "summary": "one-line description of the subject",
  "sections": [{"h2": "heading", "body": "markdown"}],
  "related": ["other subjects a fact explicitly connects to this one"],
  "disagreements": ["new fact X conflicts with existing claim Y"]
}
Sections to write (skip any with nothing to say):
- "Overview": what it is and why it matters, 2-5 sentences.
- "Key facts": bullets of durable facts (numbers, results, decisions, people involved).
- "Timeline": chronological bullets "- **<date or week>** - what happened", oldest first, when facts are dated.
- "Open items": unfinished TODOs / next steps that no later fact marks done.
Rules: rewrite into a coherent page, don't just append; keep every still-valid existing fact;
use the GLOSSARY meanings for acronyms; mention other subjects as [[Name]];
link ([[Name]] or "related") ONLY subjects named in the same fact as this one (who worked on it, what it uses,
what it is part of) - never link just because both are about AI, SLMs, or appear in the same report;
a later status update is NOT a disagreement - put it in the Timeline; only record a disagreement
when two facts cannot both be true; no Sources/Related/Disagreements sections in "sections"."""


def chunk_markdown(text: str, size: int) -> list[str]:
    """Split on headings, pack up to `size` chars; hard-split oversized sections."""
    chunks: list[str] = []
    cur = ""
    for part in re.split(r"(?m)^(?=#{1,3} )", text):
        if cur and len(cur) + len(part) > size:
            chunks.append(cur)
            cur = ""
        cur += part
        while len(cur) > size:
            chunks.append(cur[:size])
            cur = cur[size:]
    if cur.strip():
        chunks.append(cur)
    return chunks


def _heading(chunk: str, i: int) -> str:
    hs = re.findall(r"(?m)^##? (.+)$", chunk)
    if len(hs) > 2:
        return f"{hs[0].strip()} … {hs[-1].strip()}"
    return " · ".join(h.strip() for h in hs) or f"Part {i}"


def raw_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def pending_raw(vault: Vault) -> list[Path]:
    """Raw files that are new or changed since their last successful ingest."""
    shas = vault.ingested_shas()
    return [p for p in vault.raw_files() if shas.get(vault.rel(p)) != raw_sha(p.read_text(encoding="utf-8", errors="replace"))]


def resolve_raw(settings: Settings, path: Path) -> Path:
    """Accept vault-, raw-, or cwd-relative paths; copy outside files into raw/."""
    cands = [path] if path.is_absolute() else [settings.vault / path, settings.raw_dir / path, Path.cwd() / path]
    src = next((c.resolve() for c in cands if c.is_file()), None)
    if src is None:
        raise FileNotFoundError(f"source not found: {path}")
    if settings.raw_dir.resolve() not in src.parents:
        dest = settings.raw_dir / src.name
        shutil.copyfile(src, dest)
        return dest
    return src


def ingest_path(settings: Settings, path: Path, *, force: bool = False) -> dict:
    vault = Vault(settings)
    vault.ensure_layout()
    raw = resolve_raw(settings, path)
    rel = vault.rel(raw)
    text = raw.read_text(encoding="utf-8", errors="replace")
    sha = raw_sha(text)
    log = Progress(settings, "ingest")

    if not force and vault.ingested_shas().get(rel) == sha:
        log("skip", f"{rel} unchanged since last ingest")
        return {"raw": rel, "skipped": True}

    body = parse_frontmatter(text)[1]
    h1 = re.search(r"(?m)^# (.+)$", body)
    chunks = chunk_markdown(body, settings.chunk_chars)
    log("start", f"{rel}: {len(text)} chars in {len(chunks)} chunk(s)")
    glossary = vault.glossary()
    gloss_hint = "\n".join(f"- {t}: {m}" for t, m, _ in glossary) or "(empty)"
    known = vault.titles()
    claims_by_section: list[tuple[str, list[str]]] = []
    items: dict[str, dict] = {}
    errors: list[str] = []
    # Chunks extracted on an earlier successful run only contribute their stored claims; their pages are already written.
    # ponytail: an edit mid-file can shift later chunk boundaries and re-extract those chunks too.
    cache_path = settings.logs_dir / "chunks.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    seen: dict[str, list] = {} if force else cache.get(rel, {})
    done: dict[str, list] = {}

    with LLMClient(settings) as llm:
        queue = [(c, raw_sha(c)) for c in chunks]
        i = 0
        while queue:
            chunk, orig = queue.pop(0)
            i += 1
            if orig in seen:
                claims_by_section += [(h, cs) for h, cs in seen[orig]]
                done[orig] = seen[orig]
                log("extract", f"chunk {i}/{i + len(queue)}: unchanged, skipped")
                continue
            heading = _heading(chunk, i)
            log("extract", f"chunk {i}/{i + len(queue)}: {heading}")
            known_hint = ", ".join(known[:200]) or "(none yet)"
            prompt = (
                f"GLOSSARY:\n{gloss_hint}\n\nKNOWN PAGES: {known_hint}\n\n"
                f"DOCUMENT: {h1.group(1).strip() if h1 else raw.stem}\n\nTEXT:\n{chunk}"
            )
            try:
                data = llm.chat_json(EXTRACT_SYS, prompt, max_tokens=8000)
            except LLMError as e:
                # JSON mode is grammar-constrained, so invalid JSON means truncation: retry as two smaller chunks.
                if len(chunk) > 1500:
                    head = re.match(r"#{1,3} .+\n", chunk)
                    queue[:0] = [
                        (p if p.startswith("#") or not head else head.group(0) + p, orig)
                        for p in chunk_markdown(chunk, len(chunk) // 2 + 1)
                    ]
                    log("retry", f"chunk {i} failed, splitting: {str(e)[:120]}")
                    continue
                errors.append(f"chunk {i} ({heading}): {e}")
                log("error", f"chunk {i} failed: {str(e)[:200]}")
                continue
            claims = [str(c) for c in data.get("claims") or [] if c]
            claims_by_section.append((heading, claims))
            done.setdefault(orig, []).append([heading, claims])
            for kind in ("entities", "concepts"):
                for it in data.get(kind) or []:
                    if not isinstance(it, dict) or not it.get("name"):
                        continue
                    name = vault.canonical(str(it["name"]), glossary)
                    slot = items.setdefault(
                        norm(name),
                        {"name": name, "kind": kind, "category": "", "summary": "", "facts": [], "hits": 0},
                    )
                    slot["hits"] += 1
                    if kind == "entities":  # category is entity-only; concepts never get one, even if the model hallucinates it
                        slot["category"] = slot["category"] or str(it.get("category") or "")
                    slot["summary"] = slot["summary"] or str(it.get("summary") or "")
                    slot["facts"] += [str(f) for f in it.get("facts") or [] if f]

        all_claims = [c for _, cs in claims_by_section for c in cs]
        log("summarize", f"{len(all_claims)} claims, {len(items)} candidate pages")
        try:
            meta = llm.chat_json(SUMMARY_SYS, "\n".join(f"- {c}" for c in all_claims[:200]), max_tokens=800)
        except LLMError as e:
            errors.append(f"summary: {e}")
            log("error", f"summary failed, using first claims: {e}")
            meta = {"summary": " ".join(all_claims[:3])}
        title = (h1.group(1).strip() if h1 else "") or str(meta.get("title") or raw.stem)

        # Source page: re-use the page already bound to this raw file, if any.
        src_page = next(
            (p for p in vault.pages(("sources",)) if vault.read(p)[0].get("raw") == rel),
            None,
        ) or vault.page_path("sources", title)
        src_related = (
            [m.group(1) for b in section_bullets(vault.read(src_page)[1], "Related") for m in WIKILINK_RE.finditer(b)]
            if seen and src_page.exists() else []
        )
        picked = sorted(items.values(), key=lambda x: (-x["hits"], -len(x["facts"])))
        picked = picked[: settings.max_pages_per_ingest]
        claims_md = "\n\n".join(
            f"### {h}\n\n" + "\n".join(f"- {c}" for c in cs) for h, cs in claims_by_section if cs
        )
        def write_source(done_sha: str) -> None:
            # raw_sha stays empty until every step succeeded, so the next ingest retries.
            src_page.write_text(
                render_page(
                    kind="sources",
                    title=src_page.stem,
                    summary=str(meta.get("summary") or ""),
                    sections=[
                        {"h2": "Raw", "body": f"[[{rel}]]"},
                        {"h2": "Key claims", "body": claims_md},
                    ],
                    related=src_related + [it["name"] for it in picked],
                    disagreements=[str(c) for c in meta.get("contradictions") or []],
                    extra={"raw": rel, "raw_sha": done_sha},
                ),
                encoding="utf-8",
            )

        write_source("")
        source = src_page.stem
        touched = [src_page]
        linkable = {norm(t) for t in vault.titles()} | {norm(it["name"]) for it in picked}
        log("page", f"wrote {vault.rel(src_page)}")

        for n, it in enumerate(picked, 1):
            path = vault.page_path(it["kind"], it["name"])
            existed = path.exists()
            old_meta, old_body = vault.read(path) if existed else ({}, "")
            log("page", f"{'update' if existed else 'create'} {vault.rel(path)} ({n}/{len(picked)})")
            try:
                upd = llm.chat_json(
                    UPDATE_SYS,
                    f"GLOSSARY:\n{gloss_hint}\n\nPAGE: {path.stem}\n\nEXISTING PAGE:\n{old_body[: settings.page_char_budget] or '(new page)'}\n\n"
                    f"NEW FACTS (from [[{source}]]):\n"
                    + "\n".join(f"- {f}" for f in [it["summary"], *it["facts"]] if f),
                    max_tokens=3000,
                )
            except LLMError as e:
                errors.append(f"{path.stem}: {e}")
                log("error", f"skipped {path.stem}: {e}")
                continue
            old_sources = [str(s).strip("[]") for s in old_meta.get("sources") or []]
            old_related = [m.group(1) for b in section_bullets(old_body, "Related") for m in WIKILINK_RE.finditer(b)]
            path.write_text(
                render_page(
                    kind=path.parent.name,
                    title=path.stem,
                    summary=str(upd.get("summary") or it["summary"]),
                    sections=[s for s in upd.get("sections") or [] if isinstance(s, dict)]
                    or [{"h2": "Notes", "body": "\n".join(f"- {f}" for f in it["facts"])}],
                    related=old_related + [str(r) for r in upd.get("related") or [] if norm(str(r)) in linkable],
                    sources=old_sources + [source],
                    disagreements=section_bullets(old_body, "Disagreements")
                    + [str(d) for d in upd.get("disagreements") or []],
                    # Re-derive from old_meta only for entities, so a concept heals on its next ingest
                    # even if a stale category was written to disk before this guard existed.
                    extra={"category": c} if it["kind"] == "entities" and (c := old_meta.get("category") or it["category"]) else None,
                ),
                encoding="utf-8",
            )
            touched.append(path)

    if not errors:
        write_source(sha)
        cache[rel] = done
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    vault.relink(touched)
    vault.delink_broken(touched)
    vault.rebuild_index()
    pages = [vault.rel(p) for p in touched]
    vault.append_log(
        "ingest",
        source,
        f"raw: `{rel}`\n\n" + "\n".join(f"- [[{p.stem}]]" for p in touched)
        + "".join(f"\n- ⚠ {e}" for e in errors),
    )
    log("done", f"{source}: {len(pages)} pages touched, {len(errors)} errors"
        + (" (will retry on next ingest)" if errors else ""))
    return {"raw": rel, "source": source, "pages": pages, "errors": errors}


def ingest_pending(settings: Settings, *, force: bool = False) -> list[dict]:
    vault = Vault(settings)
    vault.ensure_layout()
    return [ingest_path(settings, p, force=force) for p in (vault.raw_files() if force else pending_raw(vault))]
