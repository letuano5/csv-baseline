"""Execute a SQL query against a SQLite file.

Copies DB to :memory: for isolation, then runs the query.

Public API:
  execute_sql(sqlite_path, sql) -> list[list] | dict
    Returns list of rows (without header) on success,
    or {"error": str} on failure.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def execute_sql(sqlite_path: Path, sql: str) -> list[list[Any]] | dict:
  """Run sql on an in-memory copy of sqlite_path.

  Mỗi lần gọi tạo connection riêng — an toàn khi chạy đa luồng.

  Returns:
    list[list[Any]] — rows on success
    {"error": str}  — on SQL error or file/connection error
  """
  if not sqlite_path.exists():
    return {"error": f"SQLite file not found: {sqlite_path}"}

  mem_conn: sqlite3.Connection | None = None
  try:
    src = sqlite3.connect(str(sqlite_path))
    try:
      mem_conn = sqlite3.connect(":memory:")
      src.backup(mem_conn)
    finally:
      src.close()

    cursor = mem_conn.execute(sql)
    rows = [list(row) for row in cursor.fetchall()]
    cursor.close()
    return rows
  except sqlite3.Error as e:
    return {"error": str(e)}
  except OSError as e:
    return {"error": str(e)}
  finally:
    if mem_conn is not None:
      mem_conn.close()
