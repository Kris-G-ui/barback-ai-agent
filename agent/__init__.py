"""BarBack — an LLM-driven inventory agent for bars and restaurants."""
from __future__ import annotations

from pathlib import Path

from .actions import Toolbox
from .audit import AuditLogger
from .core import Agent, TurnResult
from .llm import LLMClient, LLMError
from .store import InventoryStore

__all__ = ["build_agent", "Agent", "TurnResult", "LLMClient", "LLMError"]


def build_agent(session_id: str, llm, base_dir: str | Path = "data") -> Agent:
    """Wire up a fresh per-session store, toolbox, audit log and agent."""
    workdir = Path(base_dir) / "sessions" / session_id
    store = InventoryStore(workdir / "inventory.db")
    toolbox = Toolbox(store, workdir)
    audit = AuditLogger(store.conn, session_id, Path(base_dir) / "logs" / "audit.jsonl")
    return Agent(llm, toolbox, audit)
