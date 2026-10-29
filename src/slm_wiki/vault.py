from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

from slm_wiki.config import Settings

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
KINDS = ("sources", "entities", "concepts", "queries")
TYPE_OF = {"sources": "source", "entities": "entity", "concepts": "concept", "queries": "query"}
_BAD_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]]')


def safe_title(title: str) -> str:
    """Filesystem-safe filename stem; [[Title]] resolves to Title.md."""
    s = re.sub(r"\s+", " ", _BAD_CHARS.sub(" ", title)).strip(" .")
    return s[:120] or "Untitled"


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Values are written as JSON (valid YAML), so json.loads reads them back."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta: dict = {}
    for line in text[4:end].splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        val = val.strip()
        try:
            meta[key.strip()] = json.loads(val)
        except json.JSONDecodeError:
            meta[key.strip()] = val
    return meta, text[end + 4 :].lstrip("\n")


def section_bullets(body: str, heading: str) -> list[str]:
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", body, re.MULTILINE | re.DOTALL)
    if not m:
        return []
    return [ln[2:].strip() for ln in m.group(1).splitlines() if ln.startswith("- ")]


def _uniq(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for i in items:
        k = norm(i)
        if i and k not in seen:
            seen.add(k)
            out.append(i)
    return out


def render_page(
    *,
    kind: str,
    title: str,
    summary: str,
    sections: list[dict],
    related: list[str] = (),
    sources: list[str] = (),
    disagreements: list[str] = (),
    extra: dict | None = None,
) -> str:
    meta = {
        "type": TYPE_OF.get(kind, kind),
        # frontmatter is plain text read by search/tree, not rendered markdown: drop any [[links]] the model added
        "summary": WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), summary),
        "updated": date.today().isoformat(),
        "sources": [f"[[{s}]]" for s in _uniq(list(sources))],
        "tags": [TYPE_OF.get(kind, kind)],
        **(extra or {}),
    }
    lines = ["---", *(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()), "---", ""]
    lines += [f"# {title}", "", f"> {summary}", ""]
    for sec in sections:
        body = str(sec.get("body") or "").strip()
        if body:
            lines += [f"## {sec.get('h2') or 'Notes'}", "", body, ""]
    for heading, items in (
        ("Disagreements", _uniq(list(disagreements))),
        ("Related", [f"[[{r}]]" for r in _uniq([r.strip("[]") for r in related]) if norm(r) != norm(title)]),
        ("Sources", [f"[[{s}]]" for s in _uniq(list(sources))]),
    ):
        if items:
            lines += [f"## {heading}", "", *(f"- {i}" for i in items), ""]
    return "\n".join(lines)


class Vault:
    def __init__(self, settings: Settings):
        self.s = settings

    def ensure_layout(self) -> None:
        for d in (self.s.raw_dir, self.s.logs_dir, *(self.s.wiki_dir / k for k in KINDS)):
            d.mkdir(parents=True, exist_ok=True)
        if not self.s.log_path.exists():
            self.s.log_path.write_text("# Wiki log\n", encoding="utf-8")
        if not self.s.index_path.exists():
            self.rebuild_index()

    def pages(self, kinds: tuple[str, ...] = KINDS) -> list[Path]:
        return sorted(p for k in kinds for p in (self.s.wiki_dir / k).glob("*.md"))

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.s.vault.resolve()).as_posix()

    def find_page(self, title: str) -> Path | None:
        """Extend-don't-create: match any existing page by normalized title."""
        n = norm(title)
        return next((p for p in self.pages() if norm(p.stem) == n), None)

    def page_path(self, kind: str, title: str) -> Path:
        return self.find_page(title) or self.s.wiki_dir / kind / f"{safe_title(title)}.md"

    def read(self, path: Path) -> tuple[dict, str]:
        return parse_frontmatter(path.read_text(encoding="utf-8"))

    def titles(self, kinds: tuple[str, ...] = ("entities", "concepts")) -> list[str]:
        return [p.stem for p in self.pages(kinds)]

    def glossary(self) -> list[tuple[str, str, list[str]]]:
        """(term, meaning, aliases) rows from the table in schema/glossary.md."""
        path = self.s.schema_dir / "glossary.md"
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 3 and cells[0] and cells[0] != "Term" and not set(cells[0]) <= set("-: "):
                rows.append((cells[0], cells[1], [a.strip() for a in cells[2].split(",") if a.strip()]))
        return rows

    def canonical(self, name: str, glossary: list[tuple[str, str, list[str]]]) -> str:
        n = norm(name)
        return next((t for t, _, aliases in glossary if n in {norm(t), *map(norm, aliases)}), name)

    def relink(self, paths: list[Path]) -> None:
        """Point [[Name]] at the real file when the model's spelling differs."""
        by_norm = {norm(p.stem): p.stem for p in self.pages()}
        for term, _, aliases in self.glossary():
            if norm(term) in by_norm:
                by_norm.update({norm(a): by_norm[norm(term)] for a in aliases})

        def fix(m: re.Match) -> str:
            target, alias = m.group(1).strip(), m.group(2)
            if target.startswith("raw/") or (self.s.wiki_dir / target).exists():
                return m.group(0)
            stem = by_norm.get(norm(target))
            if not stem or stem == target:
                return m.group(0)
            return f"[[{stem}|{alias or target}]]"

        for p in paths:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                new = WIKILINK_RE.sub(fix, text)
                if new != text:
                    p.write_text(new, encoding="utf-8")

    def delink_broken(self, paths: list[Path]) -> None:
        """Drop the [[brackets]] (keep the words) around links to names that never became a page.

        Body text isn't checked against known pages the way the 'related' field is (see ingest.py),
        so the model can wikilink any name it mentions in passing. Call after relink().
        """
        titles = {norm(p.stem) for p in self.pages()}

        def fix(m: re.Match) -> str:
            target, alias = m.group(1).strip(), m.group(2)
            if target.startswith("raw/") or norm(target) in titles:
                return m.group(0)
            return alias or target

        for p in paths:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                new = WIKILINK_RE.sub(fix, text)
                if new != text:
                    p.write_text(new, encoding="utf-8")

    def rebuild_index(self) -> None:
        lines = ["# Wiki index", ""]
        for kind in KINDS:
            lines += [f"## {kind.title()}", ""]
            for p in self.pages((kind,)):
                summary = str(self.read(p)[0].get("summary") or "").replace("\n", " ")
                lines.append(f"- [[{p.stem}]] — {summary[:200]}")
            lines.append("")
        self.s.index_path.write_text("\n".join(lines), encoding="utf-8")

    def append_log(self, op: str, title: str, detail: str = "") -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{stamp}] {op} | {title}\n" + (f"\n{detail}\n" if detail else "")
        with self.s.log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    def raw_files(self) -> list[Path]:
        exts = {".md", ".txt", ".markdown"}
        return sorted(p for p in self.s.raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)

    def ingested_shas(self) -> dict[str, str]:
        """raw rel path -> sha recorded on its source page."""
        out = {}
        for p in self.pages(("sources",)):
            meta = self.read(p)[0]
            if meta.get("raw"):
                out[str(meta["raw"])] = str(meta.get("raw_sha") or "")
        return out
