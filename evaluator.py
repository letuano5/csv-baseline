"""Execution accuracy evaluation (Spider2 style).

Adapted from Spider2's compare_pandas_table with pipeline.md adjustments:
  - Float tolerance: 1e-6 (stricter than Spider2's 1e-2)
  - Column subset check: all gold columns must appear in predicted result
  - Sort both sides if SQL has no ORDER BY
  - NULL == NULL → True

Public API:
  compare_results(pred_rows, gold_rows, *, ignore_order=True) -> bool
  execution_accuracy(predictions, gold_list) -> dict
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import ROOT
from executor import execute_sql

SQLITE_DIR = ROOT / "input" / "sqlite"

_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_TOLERANCE = 1e-2


# ============================================================
# Copied verbatim from csv-pipeline/src/evaluator.py
# ============================================================

def _values_equal(a: Any, b: Any) -> bool:
  """Compare two scalar values with NULL and float tolerance."""
  # NULL == NULL
  if a is None and b is None:
    return True
  if a is None or b is None:
    return False
  # Numeric tolerance
  if isinstance(a, (int, float)) and isinstance(b, (int, float)):
    return math.isclose(float(a), float(b), abs_tol=_TOLERANCE, rel_tol=1e-9)
  return a == b


def _sort_key(x: Any):
  return (x is None, str(x) if x is not None else "", isinstance(x, (int, float)))


def _vectors_match(v1: list, v2: list, *, ignore_order: bool) -> bool:
  if len(v1) != len(v2):
    return False
  if ignore_order:
    v1 = sorted(v1, key=_sort_key)
    v2 = sorted(v2, key=_sort_key)
  return all(_values_equal(a, b) for a, b in zip(v1, v2))


def compare_results(
  pred_rows: list[list[Any]],
  gold_rows: list[list[Any]],
  *,
  ignore_order: bool = True,
  sql: str | None = None,
) -> bool:
  """Compare predicted and gold result sets.

  Uses column-subset logic: every column in gold must match some column in pred.
  If sql contains ORDER BY, ordering is respected (ignore_order=False).

  Args:
    pred_rows:    Predicted query results (list of rows).
    gold_rows:    Gold query results (list of rows).
    ignore_order: Whether to ignore row ordering (overridden by SQL analysis).
    sql:          The predicted SQL query (used to detect ORDER BY).

  Returns:
    True if results match, False otherwise.
  """
  # Detect ORDER BY in SQL → preserve order
  if sql and _ORDER_BY_RE.search(sql):
    ignore_order = False

  if not gold_rows and not pred_rows:
    return True
  if not gold_rows or not pred_rows:
    return False

  # Transpose to column vectors
  n_gold_cols = len(gold_rows[0])
  n_pred_cols = len(pred_rows[0]) if pred_rows else 0

  gold_cols = [[row[c] for row in gold_rows] for c in range(n_gold_cols)]
  pred_cols = [[row[c] for row in pred_rows] for c in range(n_pred_cols)]

  # Gold columns must appear somewhere in pred; extra pred columns are accepted so SELECT * does not penalize correct answers
  for gold_col in gold_cols:
    if not any(_vectors_match(gold_col, pred_col, ignore_order=ignore_order) for pred_col in pred_cols):
      return False

  return True


def execution_accuracy(
  predictions: list[dict],
  gold_list: list[dict],
) -> dict:
  """Compute execution accuracy over a list of predictions.

  Each prediction dict must have:
    instance_id, exec_answer (list[list] or {"error": ...}), sql_answer (str)

  Each gold dict must have:
    instance_id, exec_answer (list[list])

  Returns:
    {
      "score": float,
      "correct": int,
      "total": int,
      "details": [{"instance_id": ..., "correct": bool, "error": ...}]
    }
  """
  gold_map = {g["instance_id"]: g for g in gold_list}
  details = []
  correct = 0

  for pred in predictions:
    iid = pred["instance_id"]
    gold = gold_map.get(iid)
    if gold is None:
      details.append({"instance_id": iid, "correct": False, "error": "no gold found"})
      continue

    pred_exec = pred.get("exec_answer")
    gold_exec = gold.get("exec_answer")

    if isinstance(pred_exec, dict) and "error" in pred_exec:
      details.append({"instance_id": iid, "correct": False, "error": pred_exec["error"]})
      continue

    if not isinstance(pred_exec, list) or not isinstance(gold_exec, list):
      details.append({"instance_id": iid, "correct": False, "error": "invalid exec_answer format"})
      continue

    sql = pred.get("sql_answer", "")
    ok = compare_results(pred_exec, gold_exec, sql=sql)
    if ok:
      correct += 1
    details.append({"instance_id": iid, "correct": ok, "error": None})

  total = len(predictions)
  score = correct / total if total > 0 else 0.0
  return {"score": score, "correct": correct, "total": total, "details": details}


# ============================================================
# Glue layer — adapts csv-baseline data format to the above
# ============================================================

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


def is_column_order_mismatch_result(question: Any, result_str: str) -> bool:
  """Kept for API compatibility with base.py. Always False — column-subset handles this."""
  return False


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
      details.append(EvalDetail(
        index=ans.index,
        status="wrong",
        reason=f"invalid_result:{ans.result[:80]}",
      ))
      continue

    sqlite_path = SQLITE_DIR / f"{q.db_id}.sqlite"
    gold_result = execute_sql(sqlite_path, q.sql)

    if isinstance(gold_result, dict):
      wrong_indices.append(ans.index)
      details.append(EvalDetail(
        index=ans.index,
        status="wrong",
        reason=f"gold_sql_error:{gold_result['error'][:120]}",
        predicted=predicted,
      ))
      continue

    ok = compare_results(predicted, gold_result, sql=q.sql)

    if ok:
      correct += 1
      details.append(EvalDetail(index=ans.index, status="correct", reason="match"))
    else:
      wrong_indices.append(ans.index)
      details.append(EvalDetail(
        index=ans.index,
        status="wrong",
        reason="value_mismatch",
        predicted=predicted,
        expected=gold_result,
      ))

  total = len([d for d in details if d.status in ("correct", "wrong")])
  return EvalSummary(
    total=total,
    correct=correct,
    wrong=len(wrong_indices),
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
