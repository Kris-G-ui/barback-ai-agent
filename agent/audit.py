"""Audit logging: every user input, LLM call, validation result, confirmation
decision and action execution is written to both the SQLite ``audit_log``
table and an append-only JSON Lines file."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .store import now

log = logging.getLogger("barback")


class AuditLogger:
    def __init__(self, conn, session_id: str, jsonl_path: str | Path):
        self.conn = conn
        self.session_id = session_id
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **detail) -> None:
        ts = now()
        blob = json.dumps(detail, default=str)
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO audit_log (ts, session_id, event, detail) VALUES (?,?,?,?)",
                    (ts, self.session_id, event, blob),
                )
        except Exception as exc:  # logging must never crash the agent
            log.warning("audit db write failed: %s", exc)
        try:
            with self.jsonl_path.open("a") as f:
                f.write(json.dumps({"ts": ts, "session": self.session_id, "event": event, **detail}, default=str) + "\n")
        except OSError as exc:
            log.warning("audit file write failed: %s", exc)
        log.info("%s %s", event, blob[:300])

    def recent(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, event, detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"ts": r[0], "event": r[1], "detail": r[2]} for r in rows]
