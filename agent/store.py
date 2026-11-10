"""SQLite inventory store for a bar / restaurant.

Each user session gets its own database file seeded with demo data, so many
people can try the deployed app at once without stepping on each other.
"""
from __future__ import annotations

import difflib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    category    TEXT NOT NULL,
    unit        TEXT NOT NULL,
    quantity    REAL NOT NULL DEFAULT 0 CHECK (quantity >= 0),
    par_level   REAL NOT NULL DEFAULT 0 CHECK (par_level >= 0),
    unit_cost   REAL NOT NULL DEFAULT 0 CHECK (unit_cost >= 0),
    supplier    TEXT,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER REFERENCES items(id) ON DELETE SET NULL,
    item_name   TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- receive | usage | count | add | delete
    qty_change  REAL NOT NULL,
    reason      TEXT,
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    session_id  TEXT,
    event       TEXT NOT NULL,
    detail      TEXT
);
"""

# name, category, unit, quantity, par, unit_cost, supplier
SEED_ITEMS = [
    ("Bud Light 1/2 bbl keg", "Beer", "keg", 3, 4, 129.00, "Valley Beverage"),
    ("Guinness 1/6 bbl keg", "Beer", "keg", 1, 2, 98.00, "Valley Beverage"),
    ("Modelo Especial 12oz", "Beer", "case", 6, 5, 31.50, "Valley Beverage"),
    ("Jameson Irish Whiskey 1L", "Spirits", "bottle", 4, 6, 33.99, "Southern Glazer's"),
    ("Tito's Vodka 1L", "Spirits", "bottle", 7, 6, 24.50, "Southern Glazer's"),
    ("Jack Daniel's 1L", "Spirits", "bottle", 2, 4, 29.75, "Southern Glazer's"),
    ("Fireball 1L", "Spirits", "bottle", 5, 3, 18.25, "Southern Glazer's"),
    ("Patron Silver 750ml", "Spirits", "bottle", 1, 3, 41.00, "Southern Glazer's"),
    ("House Cabernet 750ml", "Wine", "bottle", 11, 12, 7.80, "Coastal Wine Co"),
    ("Limes", "Bar Produce", "case", 0.5, 1, 38.00, "Sysco"),
    ("Burger Patties 1/3 lb", "Food", "case", 2, 3, 64.00, "Sysco"),
    ("Chicken Wings", "Food", "lb", 18, 40, 3.10, "Sysco"),
    ("Fries 5lb bag", "Food", "bag", 9, 8, 11.40, "Sysco"),
    ("Cocktail Napkins", "Supplies", "pack", 14, 10, 4.25, "Restaurant Depot"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InventoryStore:
    def __init__(self, db_path: str | Path, seed: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.db_path.exists()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        if new and seed:
            self.seed()

    # ---------------------------------------------------------------- setup
    def seed(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM items")
            self.conn.execute("DELETE FROM transactions")
            self.conn.executemany(
                "INSERT INTO items (name, category, unit, quantity, par_level, unit_cost, supplier, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(*row, now()) for row in SEED_ITEMS],
            )

    # -------------------------------------------------------------- lookups
    def all_items(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM items ORDER BY category, name").fetchall()
        return [dict(r) for r in rows]

    def item_names(self) -> list[str]:
        return [r["name"] for r in self.conn.execute("SELECT name FROM items ORDER BY name")]

    def resolve(self, name: str) -> dict:
        """Find one item by (fuzzy) name, or raise LookupError with suggestions."""
        name = (name or "").strip()
        row = self.conn.execute("SELECT * FROM items WHERE name = ?", (name,)).fetchone()
        if row:
            return dict(row)
        names = self.item_names()
        lowered = name.lower()
        contains = [n for n in names if lowered in n.lower()]
        if len(contains) == 1:
            return self.resolve(contains[0])
        if len(contains) > 1:
            raise LookupError(f"'{name}' is ambiguous. Matches: {', '.join(contains)}")
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        if not close:  # try matching individual words, e.g. "jamison" -> "Jameson ..."
            for n in names:
                for word in n.split():
                    if difflib.SequenceMatcher(None, lowered, word.lower()).ratio() > 0.75:
                        close.append(n)
                        break
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise LookupError(f"No item named '{name}'.{hint}")

    # -------------------------------------------------------------- writes
    def log_txn(self, item: dict, kind: str, qty: float, reason: str | None) -> None:
        self.conn.execute(
            "INSERT INTO transactions (item_id, item_name, kind, qty_change, reason, ts) VALUES (?,?,?,?,?,?)",
            (item.get("id"), item["name"], kind, qty, reason, now()),
        )

    def change_quantity(self, name: str, delta: float, kind: str, reason: str | None = None) -> dict:
        item = self.resolve(name)
        new_qty = round(item["quantity"] + delta, 3)
        if new_qty < 0:
            raise ValueError(
                f"Can't remove {abs(delta)} {item['unit']} of {item['name']}: only {item['quantity']} on hand."
            )
        with self.conn:
            self.conn.execute(
                "UPDATE items SET quantity = ?, updated_at = ? WHERE id = ?", (new_qty, now(), item["id"])
            )
            self.log_txn(item, kind, delta, reason)
        return {**item, "quantity": new_qty, "previous_quantity": item["quantity"]}

    def close(self) -> None:
        self.conn.close()
