"""
Evaluate model outputs against SQL ground truth from questions.json.

Merged strengths from the simple pipeline evaluator and current baseline:
- Numeric comparison: round to 2 decimals (project rule).
- NULL/NaN normalization: NULL == NULL.
- ORDER BY awareness: keep row order strict when SQL has ORDER BY.
- Order-insensitive fallback only when SQL has no ORDER BY.
- Detect column-order mismatch explicitly for retry/diagnostics.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import re
from typing import Any

import pandas as pd

from config import get_csv_meta


@dataclass
class EvalDetail:
  index: int
  status: str
  reason: str
  expected: list[list[Any]] | None = None
  predicted: list[list[Any]] | None = None


@dataclass
class EvalSummary:
  total: int
  correct: int
  wrong: int
  error_outputs: int
  wrong_indices: list[int]
  details: list[EvalDetail]


_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _split_top_level_csv(expr: str) -> list[str]:
  parts: list[str] = []
  buf: list[str] = []
  depth = 0
  for ch in expr:
    if ch == "(":
      depth += 1
      buf.append(ch)
      continue
    if ch == ")":
      depth = max(0, depth - 1)
      buf.append(ch)
      continue
    if ch == "," and depth == 0:
      part = "".join(buf).strip()
      if part:
        parts.append(part)
      buf = []
      continue
    buf.append(ch)
  tail = "".join(buf).strip()
  if tail:
    parts.append(tail)
  return parts


def _normalize_sql_expr(expr: str) -> str:
  text = re.sub(r"\s+", " ", (expr or "").strip().rstrip(";"))
  text = re.sub(r"\bASC\b|\bDESC\b", "", text, flags=re.IGNORECASE).strip()
  return text.lower()


def _extract_select_items(sql: str) -> list[str]:
  m = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql or "", flags=re.IGNORECASE | re.DOTALL)
  if not m:
    return []
  return _split_top_level_csv(m.group(1))


def _extract_order_by_items(sql: str) -> list[str]:
  m = re.search(
    r"\bORDER\s+BY\b(.*?)(?:\bLIMIT\b|\bOFFSET\b|;|$)",
    sql or "",
    flags=re.IGNORECASE | re.DOTALL,
  )
  if not m:
    return []
  return _split_top_level_csv(m.group(1))


def _select_item_alias(item: str) -> str | None:
  text = re.sub(r"\s+", " ", item.strip())
  m = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", text, flags=re.IGNORECASE)
  if m:
    return m.group(1).lower()
  m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", text)
  if m and "." not in text:
    return m.group(1).lower()
  return None


def _order_key_positions(sql: str) -> list[int] | None:
  select_items = _extract_select_items(sql)
  order_items = _extract_order_by_items(sql)
  if not select_items or not order_items:
    return None

  normalized_select = [_normalize_sql_expr(s) for s in select_items]
  alias_to_idx: dict[str, int] = {}
  for idx, item in enumerate(select_items):
    alias = _select_item_alias(item)
    if alias:
      alias_to_idx[alias] = idx

  positions: list[int] = []
  for raw_order_item in order_items:
    cleaned = raw_order_item.strip()
    cleaned = re.sub(r"\s+(ASC|DESC)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.isdigit():
      pos = int(cleaned) - 1
      if pos < 0 or pos >= len(select_items):
        return None
      positions.append(pos)
      continue
    key = _normalize_sql_expr(cleaned)
    if key in alias_to_idx:
      positions.append(alias_to_idx[key])
      continue
    if key in normalized_select:
      positions.append(normalized_select.index(key))
      continue
    return None
  return positions


def _order_keys_sequence(rows: list[list[Any]], key_positions: list[int]) -> list[tuple[Any, ...]] | None:
  seq: list[tuple[Any, ...]] = []
  for row in rows:
    if any(pos >= len(row) for pos in key_positions):
      return None
    key: list[Any] = []
    for pos in key_positions:
      v = row[pos]
      num = _try_float(v)
      key.append(("num", round(num, 2)) if num is not None else ("txt", str(v)))
    seq.append(tuple(key))
  return seq


def _order_by_tie_aware_equal(sql: str, pred: list[list[Any]], exp: list[list[Any]]) -> bool:
  key_positions = _order_key_positions(sql)
  if not key_positions:
    return False
  pred_seq = _order_keys_sequence(pred, key_positions)
  exp_seq = _order_keys_sequence(exp, key_positions)
  if pred_seq is None or exp_seq is None:
    return False
  # Must keep the same ORDER BY key sequence; only row swaps inside tie groups are allowed.
  return pred_seq == exp_seq


def _try_float(value: Any) -> float | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
      return None
    return float(value)
  if isinstance(value, str):
    text = value.strip()
    if not text:
      return None
    text = text.replace(",", ".")
    try:
      parsed = float(text)
      if math.isnan(parsed) or math.isinf(parsed):
        return None
      return parsed
    except ValueError:
      return None
  return None


def _cell_equal(pred: Any, exp: Any) -> bool:
  if pred is None and exp is None:
    return True
  if pred is None or exp is None:
    return False
  pred_num = _try_float(pred)
  exp_num = _try_float(exp)
  if pred_num is not None and exp_num is not None:
    return round(pred_num, 2) == round(exp_num, 2)
  return str(pred) == str(exp)


def _rows_equal(pred: list[Any], exp: list[Any]) -> bool:
  if len(pred) != len(exp):
    return False
  return all(_cell_equal(p, e) for p, e in zip(pred, exp))


def _result_equal(pred: list[list[Any]], exp: list[list[Any]], *, ignore_row_order: bool) -> bool:
  if len(pred) != len(exp):
    return False
  if any(len(p) != len(e) for p, e in zip(pred, exp)):
    return False
  # First, keep strict ordered comparison.
  if all(_rows_equal(p_row, e_row) for p_row, e_row in zip(pred, exp)):
    return True

  if not ignore_row_order:
    return False

  # Fallback: order-insensitive multiset comparison.
  # This prevents false negatives when SQL ORDER BY is non-deterministic on ties
  # (same sort key values can be returned in different row orders).
  def _norm_cell(v: Any) -> Any:
    num = _try_float(v)
    if num is not None:
      return ("num", round(num, 2))
    return ("txt", str(v))

  def _norm_row(row: list[Any]) -> tuple[Any, ...]:
    return tuple(_norm_cell(v) for v in row)

  pred_counter = Counter(_norm_row(r) for r in pred)
  exp_counter = Counter(_norm_row(r) for r in exp)
  return pred_counter == exp_counter


def _is_column_order_mismatch(pred: list[list[Any]], exp: list[list[Any]]) -> bool:
  """
  Return True when predicted values match expected under a non-identity
  column permutation (i.e. same data, wrong column order).
  """
  if not pred or not exp:
    return False
  if len(pred) != len(exp):
    return False
  width = len(exp[0])
  if width <= 1:
    return False
  if any(len(r) != width for r in exp) or any(len(r) != width for r in pred):
    return False

  def _norm_cell(v: Any) -> Any:
    num = _try_float(v)
    if num is not None:
      return ("num", round(num, 2))
    return ("txt", str(v))

  exp_counter = Counter(tuple(_norm_cell(v) for v in row) for row in exp)
  identity = tuple(range(width))

  for perm in permutations(range(width)):
    if perm == identity:
      continue
    pred_perm_counter = Counter(
      tuple(_norm_cell(row[i]) for i in perm) for row in pred
    )
    if pred_perm_counter == exp_counter:
      return True
  return False


def _normalize_sql_result(rows: list[tuple[Any, ...]]) -> list[list[Any]]:
  normalized: list[list[Any]] = []
  for row in rows:
    normalized.append([None if pd.isna(v) else v for v in row])
  return normalized


def _build_conn_for_question(q: Any) -> sqlite3.Connection:
  meta = get_csv_meta(q.db_id)
  df = pd.read_csv(
    meta.path,
    encoding=meta.encoding,
    sep=meta.delimiter,
    # Keep natural dtypes so SQL predicates behave as expected:
    # - numeric comparisons like capacity_mw > 100
    # - NULL checks like latitude IS NULL
    # Reading everything as str can produce false mismatches.
  )
  conn = sqlite3.connect(":memory:")
  conn.create_function("TANH", 1, lambda x: math.tanh(float(x)) if x not in (None, "") else None)
  conn.create_function("SINH", 1, lambda x: math.sinh(float(x)) if x not in (None, "") else None)
  conn.create_function("COSH", 1, lambda x: math.cosh(float(x)) if x not in (None, "") else None)
  conn.create_function("CEIL", 1, lambda x: math.ceil(float(x)) if x not in (None, "") else None)
  conn.create_function("FLOOR", 1, lambda x: math.floor(float(x)) if x not in (None, "") else None)
  df.to_sql(q.db_id, conn, index=False, if_exists="replace")
  return conn


def _expected_from_sql(q: Any) -> list[list[Any]]:
  conn = _build_conn_for_question(q)
  try:
    cur = conn.execute(q.sql)
    rows = cur.fetchall()
    return _normalize_sql_result(rows)
  finally:
    conn.close()


def _parse_predicted(result: str) -> list[list[Any]] | None:
  if result.startswith("ERROR:"):
    return None
  try:
    parsed = json.loads(result)
  except json.JSONDecodeError:
    return None
  if not isinstance(parsed, list):
    return None
  return [row if isinstance(row, list) else [row] for row in parsed]


def is_column_order_mismatch_result(question: Any, result_str: str) -> bool:
  """
  True if `result_str` appears to contain correct values but wrong column order.
  """
  predicted = _parse_predicted(result_str)
  if predicted is None:
    return False
  try:
    expected = _expected_from_sql(question)
  except Exception:
    return False
  return _is_column_order_mismatch(predicted, expected)


def _ignore_row_order_for_sql(sql: str) -> bool:
  # If SQL explicitly requests ordering, preserve row order in comparison.
  return not _ORDER_BY_RE.search(sql or "")


def evaluate_results(results: list[Any], questions: list[Any]) -> EvalSummary:
  by_index = {q.index: q for q in questions}
  details: list[EvalDetail] = []
  wrong_indices: list[int] = []
  correct = 0
  error_outputs = 0

  for ans in sorted(results, key=lambda x: x.index):
    q = by_index.get(ans.index)
    if q is None:
      continue

    predicted = _parse_predicted(ans.result)
    if predicted is None:
      error_outputs += 1
      wrong_indices.append(ans.index)
      details.append(
        EvalDetail(
          index=ans.index,
          status="wrong",
          reason=f"invalid_result:{ans.result[:80]}",
          predicted=None,
          expected=None,
        )
      )
      continue

    try:
      expected = _expected_from_sql(q)
    except Exception as exc:  # noqa: BLE001
      wrong_indices.append(ans.index)
      details.append(
        EvalDetail(
          index=ans.index,
          status="wrong",
          reason=f"sql_eval_error:{type(exc).__name__}:{exc}",
          predicted=predicted,
          expected=None,
        )
      )
      continue

    ignore_row_order = _ignore_row_order_for_sql(getattr(q, "sql", ""))
    is_equal = _result_equal(predicted, expected, ignore_row_order=ignore_row_order)
    if (not is_equal) and (not ignore_row_order):
      # ORDER BY exists: allow row permutation only when ORDER BY key sequence
      # remains identical (swap within deterministic ties only).
      if _result_equal(predicted, expected, ignore_row_order=True):
        is_equal = _order_by_tie_aware_equal(getattr(q, "sql", ""), predicted, expected)

    if is_equal:
      correct += 1
      details.append(
        EvalDetail(
          index=ans.index,
          status="correct",
          reason="match_unordered" if ignore_row_order else "match_ordered",
        )
      )
    else:
      reason = "column_order_mismatch" if _is_column_order_mismatch(predicted, expected) else "value_mismatch"
      wrong_indices.append(ans.index)
      details.append(
        EvalDetail(
          index=ans.index,
          status="wrong",
          reason=reason,
          predicted=predicted,
          expected=expected,
        )
      )

  total = len([d for d in details if d.status in ("correct", "wrong")])
  wrong = len(wrong_indices)
  return EvalSummary(
    total=total,
    correct=correct,
    wrong=wrong,
    error_outputs=error_outputs,
    wrong_indices=wrong_indices,
    details=details,
  )


def print_eval_report(summary: EvalSummary, max_preview: int = 10) -> None:
  print("\n=== Evaluation Report ===")
  print(f"Total checked : {summary.total}")
  print(f"Correct       : {summary.correct}")
  print(f"Wrong         : {summary.wrong}")
  print(f"Error outputs : {summary.error_outputs}")
  print(f"Accuracy      : {(summary.correct / summary.total * 100.0 if summary.total else 0.0):.2f}%")

  if summary.wrong_indices:
    print(f"Wrong indices : {summary.wrong_indices}")
    print("Wrong details:")
    shown = 0
    for d in summary.details:
      if d.status != "wrong":
        continue
      print(f"- index={d.index} reason={d.reason}")
      if d.expected is not None:
        print(f"  expected={d.expected[:3]}")
      if d.predicted is not None:
        print(f"  predicted={d.predicted[:3]}")
      shown += 1
      if shown >= max_preview and len(summary.wrong_indices) > max_preview:
        print(f"  ... and {len(summary.wrong_indices) - max_preview} more wrong items")
        break


def save_eval_report(path: Path, summary: EvalSummary) -> None:
  payload = {
    "total": summary.total,
    "correct": summary.correct,
    "wrong": summary.wrong,
    "error_outputs": summary.error_outputs,
    "accuracy_percent": (summary.correct / summary.total * 100.0 if summary.total else 0.0),
    "wrong_indices": summary.wrong_indices,
    "details": [
      {
        "index": d.index,
        "status": d.status,
        "reason": d.reason,
        "expected": d.expected,
        "predicted": d.predicted,
      }
      for d in summary.details
    ],
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
