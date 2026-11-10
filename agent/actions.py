"""Action registry + executor.

Every operation the agent can perform is declared here with:
  * a parameter spec (used both to build the prompt and to validate LLM output)
  * a risk level that decides whether the user must confirm first
  * a handler that does the real work

Domains covered: database operations (SQLite inventory), file operations
(exports / reports in a sandboxed folder), system operations (database backup,
reset), and outbound API calls (webhook notifications).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

from .store import InventoryStore, now

# Risk levels -----------------------------------------------------------------
READ = "read"                # no side effects
WRITE = "write"              # normal, reversible-ish change
DESTRUCTIVE = "destructive"  # deletes / overwrites data -> confirm
EXTERNAL = "external"        # leaves the app (network) -> confirm
NEEDS_CONFIRMATION = {DESTRUCTIVE, EXTERNAL}

MAX_QTY = 10_000
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,80}$")


@dataclass
class Param:
    type: str                 # "string" | "number" | "boolean"
    description: str
    required: bool = True
    enum: list[str] | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class Action:
    name: str
    description: str
    risk: str
    params: dict[str, Param]
    handler: Callable[..., Any]
    example: dict = field(default_factory=dict)

    def spec_line(self) -> str:
        ps = []
        for pname, p in self.params.items():
            bits = [p.type]
            if p.enum:
                bits.append("one of " + "|".join(p.enum))
            if not p.required:
                bits.append("optional")
            ps.append(f"{pname} ({', '.join(bits)}): {p.description}")
        params = "; ".join(ps) if ps else "no params"
        flag = " [REQUIRES USER CONFIRMATION]" if self.risk in NEEDS_CONFIRMATION else ""
        return f"- {self.name} [{self.risk}]{flag}: {self.description}\n    params: {params}"


class ActionError(Exception):
    """Raised by handlers for expected, user-facing failures."""


class Toolbox:
    """Holds the per-session resources and the registry of actions."""

    def __init__(self, store: InventoryStore, workdir: Path, webhook_url: str | None = None):
        self.store = store
        self.workdir = Path(workdir)
        self.exports = self.workdir / "exports"
        self.backups = self.workdir / "backups"
        self.exports.mkdir(parents=True, exist_ok=True)
        self.backups.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url if webhook_url is not None else os.getenv("WEBHOOK_URL")
        self.actions: dict[str, Action] = {}
        self._register_all()

    # ----------------------------------------------------------- registry
    def register(self, name, description, risk, params, handler, example=None):
        self.actions[name] = Action(name, description, risk, params, handler, example or {})

    def catalog(self) -> str:
        return "\n".join(a.spec_line() for a in self.actions.values())

    def execute(self, name: str, params: dict) -> Any:
        action = self.actions[name]
        try:
            return action.handler(**params)
        except (LookupError, ValueError, ActionError, sqlite3.Error, OSError, requests.RequestException) as exc:
            raise ActionError(str(exc)) from exc

    # ----------------------------------------------------------- helpers
    def _safe_path(self, folder: Path, filename: str) -> Path:
        if not SAFE_FILENAME.match(filename or "") or ".." in filename:
            raise ActionError(
                f"Unsafe filename '{filename}'. Use letters, numbers, '-', '_' and '.' only."
            )
        path = (folder / filename).resolve()
        if folder.resolve() not in path.parents:
            raise ActionError("Path escapes the sandbox folder.")
        return path

    @staticmethod
    def _fmt_item(i: dict) -> dict:
        return {
            "name": i["name"],
            "category": i["category"],
            "on_hand": i["quantity"],
            "unit": i["unit"],
            "par": i["par_level"],
            "unit_cost": i["unit_cost"],
            "supplier": i["supplier"],
        }

    # =================================================== DATABASE ACTIONS
    def list_items(self, category: str | None = None, below_par_only: bool = False):
        items = self.store.all_items()
        if category:
            items = [i for i in items if i["category"].lower() == category.lower()]
        if below_par_only:
            items = [i for i in items if i["quantity"] < i["par_level"]]
        return [self._fmt_item(i) for i in items]

    def get_item(self, name: str):
        return self._fmt_item(self.store.resolve(name))

    def add_item(self, name, category, unit, quantity=0, par_level=0, unit_cost=0, supplier=None):
        try:
            self.store.resolve(name)
        except LookupError:
            pass
        else:
            raise ActionError(f"'{name}' already exists. Use receive_stock or update_item instead.")
        with self.store.conn:
            cur = self.store.conn.execute(
                "INSERT INTO items (name, category, unit, quantity, par_level, unit_cost, supplier, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (name.strip(), category, unit, quantity, par_level, unit_cost, supplier, now()),
            )
            self.store.log_txn({"id": cur.lastrowid, "name": name}, "add", quantity, "new item")
        return {"added": name, "quantity": quantity, "unit": unit}

    def receive_stock(self, name: str, quantity: float, unit_cost: float | None = None):
        item = self.store.change_quantity(name, quantity, "receive", "delivery")
        if unit_cost is not None:
            with self.store.conn:
                self.store.conn.execute("UPDATE items SET unit_cost=? WHERE name=?", (unit_cost, item["name"]))
        return {"item": item["name"], "was": item["previous_quantity"], "now": item["quantity"], "unit": item["unit"]}

    def record_usage(self, name: str, quantity: float, reason: str = "sold"):
        item = self.store.change_quantity(name, -quantity, "usage", reason)
        warning = None
        if item["quantity"] < item["par_level"]:
            warning = f"{item['name']} is now below par ({item['quantity']} < {item['par_level']})."
        return {
            "item": item["name"],
            "was": item["previous_quantity"],
            "now": item["quantity"],
            "unit": item["unit"],
            "reason": reason,
            "warning": warning,
        }

    def set_count(self, name: str, quantity: float, note: str | None = None):
        item = self.store.resolve(name)
        delta = round(quantity - item["quantity"], 3)
        updated = self.store.change_quantity(item["name"], delta, "count", note or "physical count")
        return {"item": updated["name"], "was": item["quantity"], "now": quantity, "variance": delta}

    def update_item(self, name: str, par_level=None, unit_cost=None, supplier=None, category=None):
        item = self.store.resolve(name)
        changes = {k: v for k, v in {
            "par_level": par_level, "unit_cost": unit_cost, "supplier": supplier, "category": category,
        }.items() if v is not None}
        if not changes:
            raise ActionError("Nothing to update — give at least one field.")
        sets = ", ".join(f"{k} = ?" for k in changes)
        with self.store.conn:
            self.store.conn.execute(
                f"UPDATE items SET {sets}, updated_at = ? WHERE id = ?", (*changes.values(), now(), item["id"])
            )
        return {"item": item["name"], "updated": changes}

    def delete_item(self, name: str):
        item = self.store.resolve(name)
        with self.store.conn:
            self.store.log_txn(item, "delete", -item["quantity"], "item removed")
            self.store.conn.execute("DELETE FROM items WHERE id = ?", (item["id"],))
        return {"deleted": item["name"], "had_on_hand": item["quantity"]}

    def low_stock_report(self):
        rows = []
        total = 0.0
        for i in self.store.all_items():
            if i["quantity"] < i["par_level"]:
                order_qty = math.ceil(i["par_level"] - i["quantity"])
                cost = round(order_qty * i["unit_cost"], 2)
                total += cost
                rows.append({
                    "name": i["name"], "on_hand": i["quantity"], "par": i["par_level"], "unit": i["unit"],
                    "suggested_order": order_qty, "est_cost": cost, "supplier": i["supplier"],
                })
        return {"items": rows, "estimated_total": round(total, 2)}

    def inventory_value(self):
        by_cat: dict[str, float] = {}
        for i in self.store.all_items():
            by_cat[i["category"]] = round(by_cat.get(i["category"], 0) + i["quantity"] * i["unit_cost"], 2)
        return {"by_category": by_cat, "total": round(sum(by_cat.values()), 2)}

    def recent_activity(self, limit: float = 10):
        rows = self.store.conn.execute(
            "SELECT ts, item_name, kind, qty_change, reason FROM transactions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def run_sql_query(self, sql: str):
        """Read-only SQL for ad-hoc questions. Multiple layers of protection."""
        cleaned = sql.strip().rstrip(";").strip()
        if ";" in cleaned:
            raise ActionError("Only a single SQL statement is allowed.")
        if not re.match(r"^(select|with)\b", cleaned, re.I):
            raise ActionError("Only SELECT queries are allowed.")
        if re.search(r"\b(insert|update|delete|drop|alter|create|replace|attach|pragma|vacuum)\b", cleaned, re.I):
            raise ActionError("Query contains a forbidden keyword.")
        ro = sqlite3.connect(f"file:{self.store.db_path}?mode=ro", uri=True)
        try:
            ro.row_factory = sqlite3.Row
            rows = ro.execute(cleaned).fetchmany(50)
            return [dict(r) for r in rows]
        finally:
            ro.close()

    # ======================================================= FILE ACTIONS
    def export_report(self, report: str, fmt: str = "csv", filename: str | None = None):
        if report == "inventory":
            rows = self.list_items()
        elif report == "low_stock":
            rows = self.low_stock_report()["items"]
        elif report == "activity":
            rows = self.recent_activity(500)
        else:  # valuation
            v = self.inventory_value()
            rows = [{"category": k, "value": val} for k, val in v["by_category"].items()]
            rows.append({"category": "TOTAL", "value": v["total"]})
        if not rows:
            raise ActionError(f"The {report} report is empty — nothing to export.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = filename or f"{report}-{stamp}.{fmt}"
        if not filename.endswith(f".{fmt}"):
            filename += f".{fmt}"
        path = self._safe_path(self.exports, filename)
        if path.exists():
            raise ActionError(f"{filename} already exists. Pick another name or delete it first.")
        headers = list(rows[0].keys())
        if fmt == "csv":
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)
            path.write_text(buf.getvalue())
        else:  # markdown
            lines = [f"# {report.replace('_', ' ').title()} report — {stamp}", "",
                     "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
            lines += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
            path.write_text("\n".join(lines) + "\n")
        return {"file": filename, "rows": len(rows), "bytes": path.stat().st_size}

    def draft_order(self, supplier: str | None = None, filename: str | None = None):
        items = self.low_stock_report()["items"]
        if supplier:
            items = [i for i in items if (i["supplier"] or "").lower() == supplier.lower()]
        if not items:
            raise ActionError("Nothing is below par" + (f" for {supplier}" if supplier else "") + ".")
        groups: dict[str, list] = {}
        for i in items:
            groups.setdefault(i["supplier"] or "Unknown supplier", []).append(i)
        parts = []
        for sup, its in groups.items():
            lines = [f"To: {sup}", f"Subject: Order request — {datetime.now():%b %d, %Y}", "",
                     "Hi,", "", "Please send the following:", ""]
            lines += [f"  - {i['suggested_order']} {i['unit']}(s) {i['name']}" for i in its]
            est = sum(i["est_cost"] for i in its)
            lines += ["", f"Estimated total: ${est:,.2f}", "", "Thanks!", "", "-" * 40, ""]
            parts.append("\n".join(lines))
        filename = filename or f"order-draft-{datetime.now():%Y%m%d-%H%M%S}.txt"
        path = self._safe_path(self.exports, filename)
        path.write_text("\n".join(parts))
        return {"file": filename, "suppliers": list(groups), "lines": len(items)}

    def list_files(self):
        return [{"file": p.name, "bytes": p.stat().st_size} for p in sorted(self.exports.iterdir()) if p.is_file()]

    def read_file(self, filename: str):
        path = self._safe_path(self.exports, filename)
        if not path.exists():
            raise ActionError(f"No file named {filename}.")
        text = path.read_text()
        return {"file": filename, "content": text[:4000], "truncated": len(text) > 4000}

    def delete_file(self, filename: str):
        path = self._safe_path(self.exports, filename)
        if not path.exists():
            raise ActionError(f"No file named {filename}.")
        path.unlink()
        return {"deleted": filename}

    # ===================================================== SYSTEM ACTIONS
    def backup_database(self):
        dest = self.backups / f"inventory-{datetime.now():%Y%m%d-%H%M%S}.db"
        backup = sqlite3.connect(dest)
        with backup:
            self.store.conn.backup(backup)
        backup.close()
        return {"backup": dest.name, "bytes": dest.stat().st_size}

    def reset_demo_data(self):
        self.backup_database()
        self.store.seed()
        return {"reset": True, "items": len(self.store.all_items()), "note": "backup taken first"}

    # ======================================================= API ACTIONS
    def send_notification(self, message: str):
        payload = {"content": message, "text": message}  # Discord uses content, Slack uses text
        if self.webhook_url:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            return {"sent": True, "channel": "webhook", "status": r.status_code}
        outbox = self.workdir / "outbox.jsonl"
        with outbox.open("a") as f:
            f.write(json.dumps({"ts": now(), "message": message}) + "\n")
        return {"sent": True, "channel": "local outbox (no WEBHOOK_URL configured)"}

    # ======================================================= REGISTRATION
    def _register_all(self):
        S, N, B = "string", "number", "boolean"
        name_p = Param(S, "exact item name from the inventory list")
        qty_p = Param(N, "amount in the item's unit", minimum=0.001, maximum=MAX_QTY)
        r = self.register

        r("list_items", "List inventory items, optionally filtered.", READ, {
            "category": Param(S, "e.g. Beer, Spirits, Wine, Food, Bar Produce, Supplies", required=False),
            "below_par_only": Param(B, "only items under par level", required=False),
        }, self.list_items)
        r("get_item", "Look up one item's stock, par, cost and supplier.", READ, {"name": name_p}, self.get_item)
        r("add_item", "Create a new inventory item.", WRITE, {
            "name": Param(S, "new item name, include size, e.g. 'Hennessy VS 750ml'"),
            "category": Param(S, "Beer, Spirits, Wine, Food, Bar Produce or Supplies"),
            "unit": Param(S, "counting unit, e.g. bottle, keg, case, lb"),
            "quantity": Param(N, "starting on-hand amount", required=False, minimum=0, maximum=MAX_QTY),
            "par_level": Param(N, "minimum to keep on hand", required=False, minimum=0, maximum=MAX_QTY),
            "unit_cost": Param(N, "cost per unit in USD", required=False, minimum=0, maximum=100_000),
            "supplier": Param(S, "supplier name", required=False),
        }, self.add_item)
        r("receive_stock", "Add stock from a delivery.", WRITE, {
            "name": name_p, "quantity": qty_p,
            "unit_cost": Param(N, "new cost per unit if it changed", required=False, minimum=0, maximum=100_000),
        }, self.receive_stock)
        r("record_usage", "Subtract stock that was sold, wasted, comped or spilled.", WRITE, {
            "name": name_p, "quantity": qty_p,
            "reason": Param(S, "why it left inventory", required=False, enum=["sold", "waste", "comp", "spill", "other"]),
        }, self.record_usage)
        r("set_count", "Overwrite on-hand amount with a physical count (records the variance).", WRITE, {
            "name": name_p,
            "quantity": Param(N, "the counted amount", minimum=0, maximum=MAX_QTY),
            "note": Param(S, "optional note", required=False),
        }, self.set_count)
        r("update_item", "Change an item's par level, cost, supplier or category.", WRITE, {
            "name": name_p,
            "par_level": Param(N, "new par", required=False, minimum=0, maximum=MAX_QTY),
            "unit_cost": Param(N, "new unit cost", required=False, minimum=0, maximum=100_000),
            "supplier": Param(S, "new supplier", required=False),
            "category": Param(S, "new category", required=False),
        }, self.update_item)
        r("delete_item", "Permanently remove an item from inventory.", DESTRUCTIVE, {"name": name_p}, self.delete_item)
        r("low_stock_report", "Items below par with suggested order quantity and cost.", READ, {}, self.low_stock_report)
        r("inventory_value", "Dollar value of stock on hand by category.", READ, {}, self.inventory_value)
        r("recent_activity", "Most recent stock movements.", READ, {
            "limit": Param(N, "how many (default 10)", required=False, minimum=1, maximum=100),
        }, self.recent_activity)
        r("run_sql_query", "Read-only SELECT against tables items(name, category, unit, quantity, par_level, "
          "unit_cost, supplier, updated_at) and transactions(item_name, kind, qty_change, reason, ts). "
          "Use only when no other action answers the question.", READ, {
              "sql": Param(S, "a single SELECT statement"),
          }, self.run_sql_query)
        r("export_report", "Save a report to a file the user can download.", WRITE, {
            "report": Param(S, "which report", enum=["inventory", "low_stock", "activity", "valuation"]),
            "fmt": Param(S, "file format", required=False, enum=["csv", "md"]),
            "filename": Param(S, "optional file name", required=False),
        }, self.export_report)
        r("draft_order", "Write supplier order email draft(s) for below-par items to a .txt file.", WRITE, {
            "supplier": Param(S, "limit to one supplier", required=False),
            "filename": Param(S, "optional file name", required=False),
        }, self.draft_order)
        r("list_files", "List exported files.", READ, {}, self.list_files)
        r("read_file", "Read an exported file.", READ, {"filename": Param(S, "file name")}, self.read_file)
        r("delete_file", "Delete an exported file.", DESTRUCTIVE, {"filename": Param(S, "file name")}, self.delete_file)
        r("backup_database", "Snapshot the inventory database.", WRITE, {}, self.backup_database)
        r("reset_demo_data", "Wipe inventory and reload the demo data (backs up first).", DESTRUCTIVE, {},
          self.reset_demo_data)
        r("send_notification", "Post a message to the team chat webhook (Discord/Slack).", EXTERNAL, {
            "message": Param(S, "message text, under 1500 chars"),
        }, self.send_notification)
