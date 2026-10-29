from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from slm_wiki.config import Settings


class Progress:
    """Live op progress: stderr + vault/logs/events.jsonl (tail -f it)."""

    def __init__(self, settings: Settings, op: str, *, quiet: bool = False):
        self.op = op
        self.quiet = quiet
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.events_path

    def __call__(self, stage: str, message: str, **extra: Any) -> None:
        event = {"ts": datetime.now().isoformat(timespec="seconds"), "op": self.op, "stage": stage,
                 "message": message, **extra}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if not self.quiet:
            print(f"[{self.op}:{stage}] {message}", file=sys.stderr, flush=True)
