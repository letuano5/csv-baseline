"""Prompt templates shared across all providers."""

import json
from pathlib import Path

from config import CsvMeta

SYSTEM_PROMPT = """\
You are a data analyst. You are given a CSV dataset and a question written in Vietnamese.

Answer the question using the provided data and tools.

OUTPUT FORMAT (required):
- Your LAST print statement must output ONLY a valid JSON array of arrays.
- Each inner array is one result row. No column headers.
- Empty result: []
- Single scalar: [[42]]
- Do NOT print a pandas DataFrame directly — always convert: df.values.tolist()
- Do NOT print anything before the final JSON result. No df.head(), no df.info(), no intermediate prints.
- Your response must contain EXACTLY ONE print statement.

BAD (do NOT do this):
    df = pd.read_csv("data.csv")
    print(df.head())        # ← WRONG
    result = df.groupby("category")["revenue"].mean().reset_index()
    print(result.values.tolist())

GOOD:
    df = pd.read_csv("data.csv")
    result = df.groupby("category")["revenue"].mean().reset_index()
    print(result.values.tolist())

---
EXAMPLE 1 — aggregation question:
Question: "Tính doanh thu trung bình theo từng danh mục sản phẩm"
Code:
    import pandas as pd
    df = pd.read_csv("data.csv")
    result = df.groupby("category")["revenue"].mean().reset_index()
    print(result.values.tolist())
Output: [["Electronics", 1250.5], ["Clothing", 430.0], ["Food", 89.75]]

EXAMPLE 2 — filter + select question:
Question: "Liệt kê tên và tuổi của những nhân viên trên 30 tuổi"
Code:
    import pandas as pd
    df = pd.read_csv("data.csv")
    result = df[df["age"] > 30][["name", "age"]]
    print(result.values.tolist())
Output: [["Alice", 35], ["Bob", 42], ["Carol", 31]]

EXAMPLE 3 — single count question:
Question: "Có bao nhiêu đơn hàng bị hủy?"
Code:
    import pandas as pd
    df = pd.read_csv("data.csv")
    count = df[df["status"] == "cancelled"].shape[0]
    print([[count]])
Output: [[17]]
---
"""

# Appended for OpenRouter (Chat Completions — no hosted code execution).
OPENROUTER_SYSTEM_SUFFIX = """\

---
IMPORTANT — OpenRouter / no code interpreter:
You do NOT have Python or any code execution tool. The full CSV text is in the user message.
Read it carefully, reason about the answer, and respond with ONLY a valid JSON array of arrays
as your final output (each inner array = one row; no column headers). Empty result: [].
Do not wrap the JSON in markdown fences. No extra text after the closing bracket.
"""

_PROFILE_DIR = Path(__file__).parent / "profiles" / "auto"
_PROFILE_CACHE: dict[str, dict] = {}


def _load_profile(db_id: str) -> dict | None:
  if db_id in _PROFILE_CACHE:
    return _PROFILE_CACHE[db_id]
  path = _PROFILE_DIR / f"{db_id}.json"
  if not path.exists():
    _PROFILE_CACHE[db_id] = None
    return None
  try:
    profile = json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError):
    _PROFILE_CACHE[db_id] = None
    return None
  _PROFILE_CACHE[db_id] = profile
  return profile


def _profile_guidelines(db_id: str) -> str:
  profile = _load_profile(db_id)
  if not profile:
    return ""

  inferred = profile.get("inferred_types", {}) or {}
  flags = profile.get("quality_flags", {}) or {}
  lines: list[str] = []

  date_cols = inferred.get("date_columns") or []
  numeric_cols = inferred.get("numeric_columns") or []
  bool_cols = inferred.get("boolean_like_columns") or []
  numeric_text_cols = inferred.get("numeric_as_text_columns") or []
  high_missing_cols = inferred.get("high_missing_columns") or []
  mixed_date_cols = flags.get("mixed_date_format_columns") or []

  if date_cols:
    lines.append(
      f"- Parse date columns before filtering/sorting: {', '.join(date_cols)}."
    )
  if numeric_cols:
    lines.append(
      f"- Cast numeric columns with pd.to_numeric(errors='coerce'): {', '.join(numeric_cols)}."
    )
  if bool_cols:
    lines.append(
      f"- Treat these as boolean-like flags (0/1): {', '.join(bool_cols)}."
    )
  if numeric_text_cols:
    lines.append(
      f"- These may look numeric but are text in many rows; convert carefully only when needed: {', '.join(numeric_text_cols)}."
    )
  if high_missing_cols:
    lines.append(
      f"- High-missing columns: {', '.join(high_missing_cols)}. Handle null/empty explicitly."
    )
  if mixed_date_cols:
    lines.append(
      f"- Mixed date formats detected in: {', '.join(mixed_date_cols)}. Normalize with pd.to_datetime(..., errors='coerce')."
    )
  if flags.get("non_comma_delimiter"):
    lines.append("- CSV may not be comma-delimited; use robust read_csv settings.")
  if flags.get("has_bom"):
    lines.append("- File may have UTF-8 BOM; handle encoding carefully when loading.")

  if not lines:
    return ""

  return "\nDATASET-AWARE RULES (must follow):\n" + "\n".join(lines) + "\n"


def build_user_prompt(
  question: str,
  db_id: str,
  external_knowledge: str,
  meta: CsvMeta,
) -> str:
  columns_str = ", ".join(meta.columns)
  profile_section = _profile_guidelines(db_id)
  knowledge_section = (
    f"\nEXTERNAL KNOWLEDGE (important — use this to interpret the question correctly):\n"
    f"{external_knowledge.strip()}\n"
    if external_knowledge and external_knowledge.strip()
    else ""
  )
  return (
    f"DATASET: {db_id}\n"
    f"DELIMITER: {meta.delimiter!r}  |  ENCODING: {meta.encoding!r}\n"
    f"COLUMNS: {columns_str}\n"
    f"{profile_section}"
    f"{knowledge_section}"
    f"\nQUESTION: {question}"
  )
