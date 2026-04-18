"""
Gemini runner — async concurrent calls with code_execution tool.

CSV bytes are sent inline as Part.from_bytes() placed BEFORE the text prompt.
This consistent prefix enables Gemini 2.5 Pro implicit context caching (free,
automatic on the model).

Questions are sorted by db_id in BaseRunner.run() before reaching here,
so questions sharing a CSV file are batched together → maximises cache hits.
"""

from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types as genai_types

from api_key_pool import ApiKeyPool
from config import CSV_REGISTRY, GEMINI_CONCURRENCY
from prompt import SYSTEM_PROMPT, build_user_prompt
from runners.base import BaseRunner, Question


class GeminiRunner(BaseRunner):
  provider = "gemini"

  def __init__(self, model_id: str, checkpoint_name: str):
    super().__init__(model_id, checkpoint_name)
    self._pool = ApiKeyPool("gemini")

  def _batch_size(self) -> int:
    # One "mini-batch" = one concurrent group
    return GEMINI_CONCURRENCY

  def _make_client(self, key: str) -> genai.Client:
    return genai.Client(api_key=key)

  # ------------------------------------------------------------------ #
  # Single async call                                                    #
  # ------------------------------------------------------------------ #

  async def _call_one(
    self,
    q: Question,
    semaphore: asyncio.Semaphore,
  ) -> tuple[int, Any]:
    async with semaphore:
      meta = CSV_REGISTRY[q.db_id]
      csv_bytes = meta.path.read_bytes()
      user_text = build_user_prompt(q.question, q.db_id, q.external_knowledge, meta)

      # Build contents: CSV bytes first (consistent prefix = better cache hits),
      # then the question-specific text.
      contents = [
        genai_types.Part.from_bytes(data=csv_bytes, mime_type="text/csv"),
        genai_types.Part.from_text(text=user_text),
      ]

      config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[genai_types.Tool(code_execution=genai_types.ToolCodeExecution())],
        max_output_tokens=4096,
      )

      while True:
        key = self._pool.get_active_key()
        client = self._make_client(key)
        try:
          response = await client.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
          )
          return q.index, response
        except Exception as exc:  # noqa: BLE001
          status = _http_status(exc)
          if status == 429 or "quota" in str(exc).lower():
            retry_after = _parse_retry_after(exc)
            self._pool.mark_rate_limited(key, retry_after)
            continue
          if status in (503, 500):
            self._pool.mark_rate_limited(key, 30)
            continue
          # Non-retryable
          return q.index, None

  # ------------------------------------------------------------------ #
  # BaseRunner interface                                                  #
  # ------------------------------------------------------------------ #

  def _process_batch(self, questions: list[Question]) -> dict[int, Any]:
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async def _run():
      tasks = [self._call_one(q, semaphore) for q in questions]
      return await asyncio.gather(*tasks, return_exceptions=True)

    pairs = asyncio.run(_run())
    results: dict[int, Any] = {}
    for q, outcome in zip(questions, pairs):
      if isinstance(outcome, Exception):
        results[q.index] = None
      else:
        idx, resp = outcome
        results[idx] = resp
    return results


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _http_status(exc: Exception) -> int | None:
  return getattr(exc, "status_code", None) or getattr(exc, "code", None)


def _parse_retry_after(exc: Exception) -> float | None:
  headers = getattr(exc, "response", None)
  if headers is not None:
    headers = getattr(headers, "headers", {}) or {}
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val:
      try:
        return float(val)
      except ValueError:
        pass
  return None
