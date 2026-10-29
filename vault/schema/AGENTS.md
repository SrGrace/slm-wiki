# Wiki schema

Contract for anyone (human or agent) touching this vault. The `slm-wiki` engine enforces it; an 8B model is never trusted to keep the graph consistent on its own.

## Layers

| Folder | Owner | Rule |
|---|---|---|
| `raw/` | you | Immutable sources. Never edited after drop-in. Changing a file → next `ingest` re-compiles it. |
| `wiki/` | engine | Generated pages. Hand edits are fine; they are merged on the next update. |
| `schema/` | you | This file + [[glossary]] (term, meaning, aliases), which is fed to every prompt. |
| `logs/` | engine | `events.jsonl` live progress (the UI's Activity tab). |

## Pages

- One file per subject, **filename = title** (`wiki/entities/CodeQL.md`), so `[[CodeQL]]` resolves.
- Kinds: `sources/` (one per raw file), `entities/`, `concepts/`, `queries/` (saved answers).
- Frontmatter: `type`, `summary`, `updated`, `sources` (links to source pages), `tags`.
  - Entities add `category` (person, project, tool, model, org, other).
  - Source pages add `raw` and `raw_sha`.
- Body: `# Title`, `> summary`, then Overview / Key facts / Timeline / Open items, then `## Disagreements`, `## Related`, `## Sources`.

## Invariants

1. Extend, don't create: an existing page with the same normalized title is always updated instead, and glossary aliases resolve to their term's page.
2. Never drop a fact silently: real conflicts go to `## Disagreements`, which is append-only. Status changes over time go to `## Timeline`.
3. Provenance accumulates: `sources` is a union across ingests.
4. Every answer cites `[[Page]]`; citations not in the retrieved context are flagged.
5. `wiki/index.md` is derived from frontmatter (`slm-wiki reindex`), `wiki/log.md` is append-only.
