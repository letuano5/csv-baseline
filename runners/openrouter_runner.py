"""
OpenRouter runner — OpenAI-compatible Chat Completions API (async concurrent).

Unlike gemini/openai/claude runners in this repo, OpenRouter path does not use
server-side code execution. We embed CSV text in the user message and force
strict JSON-only output via prompt instructions.

Design (same style as other runners):
1) Async concurrency with one mini-batch.
2) API key rotation using ApiKeyPool on rate-limit / transient failures.
3) Retry with timeout per request and non-retryable fast-fail.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any
from types import SimpleNamespace

import openai
from openai import AsyncOpenAI

from api_key_pool import ApiKeyPool
from config import CSV_REGISTRY, OPENROUTER_CONCURRENCY
from prompt import OPENROUTER_SYSTEM_SUFFIX, SYSTEM_PROMPT, build_user_prompt
from runners.base import BaseRunner, Question

_MAX_TOKENS = 4096
_REQUEST_TIMEOUT_SECONDS = 180
_MAX_ATTEMPTS_PER_QUESTION = 8
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_LOCAL_EXEC_TIMEOUT_SECONDS = 40

_PY_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


class OpenRouterRunner(BaseRunner):
  provider = "openrouter"

  def __init__(self, model_id: str, checkpoint_name: str, max_rows: int | None = None):
    super().__init__(model_id, checkpoint_name, max_rows)
    self._pool = ApiKeyPool("openrouter")
    self._csv_cache: dict[str, str] = {}
    self._csv_cache_full: dict[str, str] = {}

  def _batch_size(self) -> int:
    # Keep mini-batch aligned with concurrency so progress/checkpoint happens
    # after each concurrent group (same behavior as Gemini runner).
    return OPENROUTER_CONCURRENCY

  @staticmethod
  def _base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/")

  @staticmethod
  def _default_headers() -> dict[str, str] | None:
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
      headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_APP_TITLE", "csv-baseline").strip()
    if title:
      headers["X-Title"] = title
    return headers or None

  def _build_client(self, key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
      api_key=key,
      base_url=self._base_url(),
      default_headers=self._default_headers(),
    )

  def _csv_text(self, db_id: str) -> str:
    """Trimmed CSV for embedding in the prompt (saves tokens)."""
    cached = self._csv_cache.get(db_id)
    if cached is not None:
      return cached
    meta = CSV_REGISTRY[db_id]
    text = meta.path.read_text(encoding=meta.encoding, errors="replace")
    if self.max_rows is not None:
      text = self._trim_csv(text, self.max_rows)
    self._csv_cache[db_id] = text
    return text

  def _csv_text_full(self, db_id: str) -> str:
    """Full CSV for local execution — always untruncated."""
    cached = self._csv_cache_full.get(db_id)
    if cached is not None:
      return cached
    meta = CSV_REGISTRY[db_id]
    text = meta.path.read_text(encoding=meta.encoding, errors="replace")
    self._csv_cache_full[db_id] = text
    return text

  def _build_messages(self, q: Question) -> list[dict[str, str]]:
    meta = CSV_REGISTRY[q.db_id]
    csv_text = self._csv_text(q.db_id)
    system_text = SYSTEM_PROMPT + OPENROUTER_SYSTEM_SUFFIX
    user_text = (
      f"<csv filename=\"{meta.filename}\">\n{csv_text}\n</csv>\n\n"
      + build_user_prompt(q.question, q.db_id, q.external_knowledge, meta, self.max_rows)
    )
    return [
      {"role": "system", "content": system_text},
      {"role": "user", "content": user_text},
    ]

  @staticmethod
  def _extract_python_code(content: str) -> str | None:
    text = (content or "").strip()
    if not text:
      return None
    m = _PY_CODE_FENCE_RE.search(text)
    if m:
      code = m.group(1).strip()
      return code or None
    # If model already returns plain code (no fences), accept it directly.
    if "\n" in text and ("import pandas" in text or "pd.read_csv" in text):
      return text
    return None

  def _execute_python_code(self, code: str, q: Question) -> str | None:
    meta = CSV_REGISTRY[q.db_id]
    csv_text = self._csv_text_full(q.db_id)
    with tempfile.TemporaryDirectory(prefix="openrouter_exec_") as tmpdir:
      tmp = tempfile.gettempdir()
      _ = tmp  # keep linter quiet on some environments
      script_path = os.path.join(tmpdir, "script.py")
      data_path = os.path.join(tmpdir, "data.csv")
      alt_name_path = os.path.join(tmpdir, meta.filename)
      try:
        with open(script_path, "w", encoding="utf-8") as fh:
          fh.write(code)
        with open(data_path, "w", encoding="utf-8") as fh:
          fh.write(csv_text)
        # Also provide original filename in case model uses meta filename.
        with open(alt_name_path, "w", encoding="utf-8") as fh:
          fh.write(csv_text)
      except OSError:
        return None

      try:
        proc = subprocess.run(
          [sys.executable, script_path],
          cwd=tmpdir,
          capture_output=True,
          text=True,
          timeout=_LOCAL_EXEC_TIMEOUT_SECONDS,
          check=False,
        )
      except (subprocess.SubprocessError, OSError):
        return None

      if proc.returncode != 0:
        return None
      stdout = (proc.stdout or "").strip()
      if not stdout:
        return None
      return stdout

  def _maybe_convert_response_to_exec_output(self, response: Any, q: Question) -> Any:
    """
    OpenRouter sometimes returns Python code instead of final JSON.
    To align behavior with other providers' code-execution flow, run that code locally
    and convert the response content into execution stdout.
    """
    try:
      content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
      return response
    if content is None:
      return response

    code = self._extract_python_code(str(content))
    if not code:
      return response

    stdout = self._execute_python_code(code, q)
    if not stdout:
      return response

    # Build a lightweight response shape compatible with extract_result_openrouter().
    return SimpleNamespace(
      choices=[SimpleNamespace(message=SimpleNamespace(content=stdout))]
    )

  async def _call_one_async(
    self,
    q: Question,
    semaphore: asyncio.Semaphore,
    clients: dict[str, AsyncOpenAI],
  ) -> tuple[int, Any]:
    async with semaphore:
      messages = self._build_messages(q)

      for _ in range(_MAX_ATTEMPTS_PER_QUESTION):
        key = await self._pool.get_active_key_async()
        client = clients[key]
        try:
          response = await asyncio.wait_for(
            client.chat.completions.create(
              model=self.model_id,
              messages=messages,
              max_tokens=_MAX_TOKENS,
              temperature=0.0,  # determinism helps strict-format compliance
            ),
            timeout=_REQUEST_TIMEOUT_SECONDS,
          )
          response = self._maybe_convert_response_to_exec_output(response, q)
          return q.index, response
        except asyncio.TimeoutError:
          # Treat timeout as transient congestion on that key/route.
          self._pool.mark_rate_limited(key, 15)
          continue
        except openai.RateLimitError as exc:
          self._pool.mark_rate_limited(key, _parse_retry_after(exc))
          continue
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
          status = getattr(exc, "status_code", None)
          if _is_retryable_status(status):
            self._pool.mark_rate_limited(key, 30 if status != 429 else _parse_retry_after(exc))
            continue
          return q.index, None
        except Exception as exc:  # noqa: BLE001
          # SDKs/providers sometimes wrap retryable errors without status codes.
          msg = str(exc).lower()
          if any(t in msg for t in ("rate limit", "temporar", "timeout", "overloaded", "busy")):
            self._pool.mark_rate_limited(key, 20)
            continue
          return q.index, None

      return q.index, None

  def _process_batch(self, questions: list[Question]) -> dict[int, Any]:
    semaphore = asyncio.Semaphore(OPENROUTER_CONCURRENCY)

    async def _run():
      clients = {key: self._build_client(key) for key in self._pool._keys}
      try:
        tasks = [self._call_one_async(q, semaphore, clients) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=True)
      finally:
        for client in clients.values():
          await client.close()

    pairs = asyncio.run(_run())
    results: dict[int, Any] = {}
    for q, outcome in zip(questions, pairs):
      if isinstance(outcome, Exception):
        results[q.index] = None
      else:
        idx, response = outcome
        results[idx] = response
    return results


def _is_retryable_status(status: int | None) -> bool:
  return status in (408, 409, 425, 429, 500, 502, 503, 504)


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
