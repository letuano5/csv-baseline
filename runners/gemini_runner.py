"""
Gemini runner — async concurrent calls with code_execution tool.

CSV bytes are sent inline as Part.from_bytes() placed BEFORE the text prompt.
This consistent prefix enables Gemini 2.5 Pro implicit context caching (free,
automatic on the model).

Questions are sorted by db_id in BaseRunner.run() before reaching here,
so questions sharing a CSV file are batched together → maximises cache hits.

Auth modes (auto-detected from .env / environment):
  Vertex AI  — set GOOGLE_GENAI_USE_VERTEXAI=true (+ GOOGLE_CLOUD_PROJECT,
               GOOGLE_CLOUD_LOCATION).  Single shared client, no API key pool.
  API key    — set GOOGLE_API_KEY (or GOOGLE_API_KEY_1/2/...).
               Uses ApiKeyPool with automatic rotation on rate-limit.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from api_key_pool import ApiKeyPool
from config import CSV_REGISTRY, GEMINI_CONCURRENCY, OUTPUT_DIR
from prompt import SYSTEM_PROMPT, build_user_prompt
from runners.base import BaseRunner, Question


# ---------------------------------------------------------------------------
# Debug response serializer
# ---------------------------------------------------------------------------

def _serialize_part(part: Any) -> dict:
  out: dict = {}
  text = getattr(part, "text", None)
  if text is not None:
    out["text"] = text
  ec = getattr(part, "executable_code", None)
  if ec is not None:
    out["executable_code"] = {
      "code": getattr(ec, "code", None),
      "language": str(getattr(ec, "language", None)),
      "id": getattr(ec, "id", None),
    }
  cer = getattr(part, "code_execution_result", None)
  if cer is not None:
    out["code_execution_result"] = {
      "outcome": str(getattr(cer, "outcome", None)),
      "output": getattr(cer, "output", None),
      "id": getattr(cer, "id", None),
    }
  return out


def _serialize_candidate(cand: Any) -> dict:
  content = getattr(cand, "content", None)
  parts = []
  if content is not None:
    for p in (getattr(content, "parts", None) or []):
      parts.append(_serialize_part(p))
  return {
    "role": getattr(content, "role", None) if content else None,
    "parts": parts,
    "finish_reason": str(getattr(cand, "finish_reason", None)),
    "token_count": getattr(cand, "token_count", None),
    "index": getattr(cand, "index", None),
  }


def _serialize_usage(usage: Any) -> dict | None:
  if usage is None:
    return None
  return {
    "prompt_token_count": getattr(usage, "prompt_token_count", None),
    "candidates_token_count": getattr(usage, "candidates_token_count", None),
    "total_token_count": getattr(usage, "total_token_count", None),
    "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
  }


def serialize_gemini_response(response: Any) -> dict:
  """Convert a GenerateContentResponse to a JSON-serializable dict."""
  candidates = getattr(response, "candidates", None) or []
  return {
    "model_version": getattr(response, "model_version", None),
    "response_id": getattr(response, "response_id", None),
    "usage_metadata": _serialize_usage(getattr(response, "usage_metadata", None)),
    "candidates": [_serialize_candidate(c) for c in candidates],
  }


def save_debug_response(response: Any, question_index: int, out_dir: Path | None = None) -> None:
  """Save a single Gemini response to JSON for inspection."""
  dest = (out_dir or OUTPUT_DIR) / "gemini_debug_responses"
  dest.mkdir(parents=True, exist_ok=True)
  payload = serialize_gemini_response(response)
  path = dest / f"response_q{question_index}.json"
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"[GeminiRunner] Debug response saved → {path}")


def _use_vertex() -> bool:
  return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in ("1", "true", "yes")


class GeminiRunner(BaseRunner):
  provider = "gemini"

  def __init__(self, model_id: str, checkpoint_name: str):
    super().__init__(model_id, checkpoint_name)
    self._vertex = _use_vertex()
    if self._vertex:
      # genai.Client() reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION /
      # GOOGLE_GENAI_USE_VERTEXAI directly from the environment (already set by
      # load_dotenv() in main.py).
      self._client: genai.Client | None = genai.Client()
      self._pool: ApiKeyPool | None = None
      self._key_clients: dict[str, genai.Client] = {}
      print(
        f"[GeminiRunner] Vertex AI mode — "
        f"project={os.getenv('GOOGLE_CLOUD_PROJECT')}, "
        f"location={os.getenv('GOOGLE_CLOUD_LOCATION')}"
      )
    else:
      self._client = None
      self._pool = ApiKeyPool("gemini")
      self._key_clients: dict[str, genai.Client] = {}
    self._csv_cache: dict[str, bytes] = {}

  def _batch_size(self) -> int:
    return GEMINI_CONCURRENCY

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
      db_id = q.db_id
      if db_id not in self._csv_cache:
        self._csv_cache[db_id] = meta.path.read_bytes()
      csv_bytes = self._csv_cache[db_id]
      user_text = build_user_prompt(q.question, q.db_id, q.external_knowledge, meta)

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
        if self._vertex:
          client = self._client
          key = None
        else:
          key = self._pool.get_active_key()
          if key not in self._key_clients:
            self._key_clients[key] = genai.Client(api_key=key)
          client = self._key_clients[key]

        try:
          response = await client.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
          )
          if os.getenv("GEMINI_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            save_debug_response(response, q.index, OUTPUT_DIR / self.checkpoint_name)
          return q.index, response
        except Exception as exc:  # noqa: BLE001
          status = _http_status(exc)
          if status == 429 or "quota" in str(exc).lower():
            if self._vertex:
              await asyncio.sleep(60)
            else:
              retry_after = _parse_retry_after(exc)
              self._pool.mark_rate_limited(key, retry_after)
            continue
          if status in (503, 500):
            if self._vertex:
              await asyncio.sleep(30)
            else:
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
