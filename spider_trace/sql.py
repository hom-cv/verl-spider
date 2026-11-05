"""Bounded read-only SQLite execution, independent of veRL and GPU libraries."""
import json
import re
import sqlite3
from pathlib import Path

# Deliberately small: no time, random, extension loading, or order-sensitive aggregation.
FUNCTIONS = set("abs avg coalesce count ifnull instr length like lower ltrim max min nullif replace round rtrim substr substring sum total trim typeof upper".split())


def database_path(root, db_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", db_id):
        raise ValueError("Invalid database ID")
    root = Path(root).resolve()
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Database not found: {db_id}")
    return path


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _authorize(action, arg1, arg2, database, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def execute_query(path, sql, max_rows=100, max_chars=12000):
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 20000:
        return {"ok": False, "error": "Expected a nonempty SQL string of at most 20000 characters"}
    conn = connect(path)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(_authorize)
        # A fixed VM instruction budget avoids time-dependent query cutoffs.
        ticks = 0
        def budget():
            nonlocal ticks
            ticks += 1
            return int(ticks > 10000)
        conn.set_progress_handler(budget, 1000)
        cursor = conn.execute(sql)  # sqlite3 rejects multiple statements.
        if cursor.description is None:
            raise ValueError("Only read queries are supported")
        rows = cursor.fetchmany(max_rows + 1)
        if any(isinstance(value, bytes) for row in rows for value in row):
            raise ValueError("Binary columns are unsupported")
        result = {"ok": True, "columns": [c[0] for c in cursor.description],
                  "rows": [list(r) for r in rows[:max_rows]], "truncated": len(rows) > max_rows}
        while len(json.dumps(result, ensure_ascii=True)) > max_chars and result["rows"]:
            result["rows"].pop()
            result["truncated"] = True
        return result
    except (sqlite3.Error, sqlite3.Warning, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()
