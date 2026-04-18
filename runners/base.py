"""
Abstract base runner — Strategy + Template Method pattern.

Subclasses implement _process_batch() which submits a mini-batch to the API
and returns {question_index: raw_response_object}.

The template method run() handles:
  - checkpoint load / skip already-answered questions
  - sorting by db_id for maximum prompt-cache hits
  - mini-batch chunking
  - checkpoint save after each batch
  - tqdm progress bar
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

import checkpointing
from config import CSV_REGISTRY, MINI_BATCH_SIZE
from result_parser import extract_result


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Question:
  index: int
  db_id: str
  sql_complexity: str
  question_style: str
  question: str
  external_knowledge: str
  cot: str
  sql: str


@dataclass
class AnsweredQuestion:
  index: int
  db_id: str
  sql_complexity: str
  question_style: str
  question: str
  external_knowledge: str
  cot: str
  sql: str
  result: str  # JSON string of list-of-lists, or "ERROR:..."
  raw_output: str = field(default="")  # raw code-execution stdout before JSON parsing


def load_questions(path) -> list[Question]:
  with open(path, encoding="utf-8") as fh:
    raw = json.load(fh)
  return [
    Question(
      index=i,
      db_id=item["db_id"],
      sql_complexity=item.get("sql_complexity", ""),
      question_style=item.get("question_style", ""),
      question=item["question"],
      external_knowledge=item.get("external_knowledge", ""),
      cot=item.get("cot", ""),
      sql=item.get("sql", ""),
    )
    for i, item in enumerate(raw)
  ]


# ---------------------------------------------------------------------------
# Base runner
# ---------------------------------------------------------------------------

class BaseRunner(ABC):
  provider: str  # must be set by subclass

  def __init__(self, model_id: str, checkpoint_name: str):
    self.model_id = model_id
    self.checkpoint_name = checkpoint_name

  # -- To be implemented by each provider --

  @abstractmethod
  def _process_batch(self, questions: list[Question]) -> dict[int, Any]:
    """
    Submit a mini-batch to the API.
    Returns {question.index: raw_response_object}.
    Missing indices are treated as errors.
    """

  def _batch_size(self) -> int:
    return MINI_BATCH_SIZE

  # -- Template method --

  def run(
    self,
    questions: list[Question],
    retry_errors: bool = False,
  ) -> list[AnsweredQuestion]:
    """
    Run all questions, skipping those already in the checkpoint.

    Args:
      retry_errors: If True, re-run questions whose saved result starts with
                    "ERROR:" so they get a fresh API call. Useful when a
                    parser bug caused spurious errors on a previous run.
    """
    done = checkpointing.answered_indices(
      self.checkpoint_name, self.model_id, retry_errors=retry_errors
    )
    existing = checkpointing.load_checkpoint(self.checkpoint_name, self.model_id)

    # Sort remaining by db_id so questions sharing the same CSV are adjacent
    # → maximises prompt-cache hits within a mini-batch
    remaining = sorted(
      [q for q in questions if q.index not in done],
      key=lambda q: q.db_id,
    )

    if not remaining:
      print(f"[{self.model_id}] All {len(questions)} questions already answered.")
      return [AnsweredQuestion(**{**{"raw_output": ""}, **r}) if isinstance(r, dict) else r for r in existing]

    print(
      f"[{self.model_id}] {len(done)} already done, "
      f"{len(remaining)} remaining. Batch size: {self._batch_size()}"
    )

    # When retrying errors, drop the old error entries from existing so they
    # get replaced with fresh results.
    if retry_errors:
      results: list[dict] = [
        r for r in existing
        if not (isinstance(r, dict) and r.get("result", "").startswith("ERROR:"))
      ]
    else:
      results = list(existing)

    batch_size = self._batch_size()

    with tqdm(total=len(remaining), desc=self.model_id, unit="q") as pbar:
      for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        try:
          raw_responses = self._process_batch(batch)
        except Exception as exc:
          print(f"\n[{self.model_id}] Batch failed: {exc}. Saving checkpoint before aborting.")
          checkpointing.save_checkpoint(self.checkpoint_name, self.model_id, results)
          raise

        for q in batch:
          raw = raw_responses.get(q.index)
          if raw is None:
            result_str, raw_output = "ERROR:no_response", ""
          else:
            result_str, raw_output = extract_result(raw, self.provider)

          answered = AnsweredQuestion(
            index=q.index,
            db_id=q.db_id,
            sql_complexity=q.sql_complexity,
            question_style=q.question_style,
            question=q.question,
            external_knowledge=q.external_knowledge,
            cot=q.cot,
            sql=q.sql,
            result=result_str,
            raw_output=raw_output,
          )
          results.append(answered)
          pbar.update(1)

        # Save after every mini-batch
        checkpointing.save_checkpoint(self.checkpoint_name, self.model_id, results)

    return [
      AnsweredQuestion(**{**{"raw_output": ""}, **r}) if isinstance(r, dict) else r
      for r in results
    ]

  # -- Utility --

  @staticmethod
  def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
      yield lst[i : i + size]
