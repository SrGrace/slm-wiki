from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from slm_wiki.config import Settings
from slm_wiki.progress import Progress
from slm_wiki.vault import WIKILINK_RE, Vault, norm, section_bullets


def lint(settings: Settings) -> dict:
    """Deterministic health check; writes wiki/Lint Report.md so it is browsable in the UI."""
    vault = Vault(settings)
    vault.ensure_layout()
    pages = vault.pages()
    by_norm = {norm(p.stem): p for p in pages}
    inbound: dict[str, set[str]] = defaultdict(set)
    broken: list[str] = []
    no_frontmatter: list[str] = []
    disputes: list[str] = []

    for p in pages:
        text = p.read_text(encoding="utf-8")
        meta, body = vault.read(p)
        if not meta:
            no_frontmatter.append(p.stem)
        if section_bullets(body, "Disagreements"):
            disputes.append(p.stem)
        for m in WIKILINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.startswith("raw/"):
                if not (settings.vault / target).exists():
                    broken.append(f"[[{p.stem}]] → `{target}`")
                continue
            hit = by_norm.get(norm(target))
            if hit is None:
                broken.append(f"[[{p.stem}]] → {target}")
            elif hit != p:
                inbound[hit.stem].add(p.stem)

    orphans = [p.stem for p in vault.pages(("entities", "concepts")) if not inbound[p.stem]]
    # ponytail: O(n²) token-overlap pass — fine for hundreds of pages.
    dupes = []
    toks = [(p.stem, set(p.stem.lower().split())) for p in pages]
    for i, (a, ta) in enumerate(toks):
        for b, tb in toks[i + 1 :]:
            if ta and tb and len(ta & tb) / len(ta | tb) >= 0.6:
                dupes.append(f"[[{a}]] ≈ [[{b}]]")

    report = {
        "pages": len(pages),
        "broken_links": broken,
        "orphans": orphans,
        "near_duplicates": dupes,
        "disagreements": disputes,
        "missing_frontmatter": no_frontmatter,
    }
    lines = ["# Lint Report", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M} · {len(pages)} pages_", ""]
    for key, label in (
        ("disagreements", "Pages with open disagreements"),
        ("broken_links", "Broken links (write the page or fix the link)"),
        ("orphans", "Orphans (nothing links here)"),
        ("near_duplicates", "Possible duplicates (merge one into the other)"),
        ("missing_frontmatter", "Missing frontmatter"),
    ):
        items = report[key]
        lines += [f"## {label} ({len(items)})", ""]
        lines += [f"- {i if '[[' in i else f'[[{i}]]'}" for i in items] or ["- none"]
        lines.append("")
    (settings.wiki_dir / "Lint Report.md").write_text("\n".join(lines), encoding="utf-8")
    vault.append_log("lint", f"{len(pages)} pages", " · ".join(f"{k}={len(v)}" for k, v in report.items() if isinstance(v, list)))
    Progress(settings, "lint", quiet=True)("done", "lint complete")
    return report
