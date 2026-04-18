"""
OpenAI runner — Responses API + code_interpreter tool + Batch API.

Strategy:
1. Upload each of the 7 CSVs once via Files API; cache file_ids by db_id.
2. Build batch JSONL targeting /v1/responses with code_interpreter tool.
3. Submit via client.batches.create(); poll until completed.
4. Fall back to async concurrent individual Responses API calls if batch
   endpoint /v1/responses is not supported (detected at runtime).

Prompt caching is automatic for OpenAI — no extra code needed.
Static content (system prompt, CSV file) is placed before variable content
to maximise cache hits.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI

from api_key_pool import ApiKeyPool
from config import CSV_REGISTRY, OUTPUT_DIR, POLL_INTERVAL_SECONDS, POLL_TIMEOUT_SECONDS
from prompt import SYSTEM_PROMPT, build_user_prompt
from runners.base import BaseRunner, Question

_MAX_TOKENS = 4096
_ASYNC_CONCURRENCY = 20
_RESPONSES_ENDPOINT = "/v1/responses"


class OpenAIRunner(BaseRunner):
  provider = "openai"

  def __init__(self, model_id: str, checkpoint_name: str):
    super().__init__(model_id, checkpoint_name)
    self._pool = ApiKeyPool("openai")
    self._file_id_cache: dict[str, str] = {}   # db_id → file_id
    self._use_batch = False

  def _get_client(self, key: str) -> OpenAI:
    return OpenAI(api_key=key)

  def _get_async_client(self, key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=key)

  # ------------------------------------------------------------------ #
  # File upload                                                          #
  # ------------------------------------------------------------------ #

  def _upload_csv(self, db_id: str) -> str:
    if db_id in self._file_id_cache:
      return self._file_id_cache[db_id]

    meta = CSV_REGISTRY[db_id]
    while True:
      key = self._pool.get_active_key()
      client = self._get_client(key)
      try:
        with open(meta.path, "rb") as fh:
          file_obj = client.files.create(file=(meta.filename, fh, "text/csv"), purpose="user_data")
        self._file_id_cache[db_id] = file_obj.id
        print(f"[OpenAI] Uploaded {meta.filename} → {file_obj.id}")
        return file_obj.id
      except openai.RateLimitError as exc:
        self._pool.mark_rate_limited(key, _parse_retry_after(exc))
      except (openai.APIStatusError, openai.APIConnectionError) as exc:
        status = getattr(exc, "status_code", None)
        if status in (503, 529):
          self._pool.mark_rate_limited(key, 30)
        else:
          raise

  def _ensure_uploads(self, questions: list[Question]) -> None:
    for db_id in {q.db_id for q in questions}:
      self._upload_csv(db_id)

  # ------------------------------------------------------------------ #
  # Request body                                                         #
  # ------------------------------------------------------------------ #

  def _build_request_body(self, q: Question) -> dict:
    file_id = self._file_id_cache[q.db_id]
    meta = CSV_REGISTRY[q.db_id]
    return {
      "model": self.model_id,
      "max_output_tokens": _MAX_TOKENS,
      "tools": [
        {
          "type": "code_interpreter",
          "container": {"type": "auto", "file_ids": [file_id]},
        }
      ],
      "instructions": SYSTEM_PROMPT,
      "input": [
        {
          "role": "user",
          "content": build_user_prompt(q.question, q.db_id, q.external_knowledge, meta),
        }
      ],
    }

  # ------------------------------------------------------------------ #
  # Batch API — pending batch persistence                               #
  # ------------------------------------------------------------------ #

  def _pending_batch_path(self) -> Path:
    return OUTPUT_DIR / self.checkpoint_name / f"{self.model_id}.pending_batch.json"

  def _save_pending_batch(self, batch_id: str, questions: list[Question]) -> None:
    path = self._pending_batch_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
      json.dumps({"batch_id": batch_id, "question_indices": [q.index for q in questions]}),
      encoding="utf-8",
    )

  def _load_pending_batch(self) -> tuple[str, list[int]] | None:
    path = self._pending_batch_path()
    if not path.exists():
      return None
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
      return data["batch_id"], data["question_indices"]
    except (KeyError, json.JSONDecodeError, OSError):
      return None

  def _clear_pending_batch(self) -> None:
    self._pending_batch_path().unlink(missing_ok=True)

  # ------------------------------------------------------------------ #
  # Batch API                                                            #
  # ------------------------------------------------------------------ #

  def _submit_batch(self, questions: list[Question]) -> str:
    lines = [
      json.dumps(
        {
          "custom_id": str(q.index),
          "method": "POST",
          "url": _RESPONSES_ENDPOINT,
          "body": self._build_request_body(q),
        },
        ensure_ascii=False,
      )
      for q in questions
    ]
    jsonl_bytes = "\n".join(lines).encode("utf-8")

    while True:
      key = self._pool.get_active_key()
      client = self._get_client(key)
      try:
        # Upload JSONL
        batch_file = client.files.create(
          file=("batch_input.jsonl", jsonl_bytes, "application/jsonl"),
          purpose="batch",
        )
        # Create batch
        batch = client.batches.create(
          input_file_id=batch_file.id,
          endpoint=_RESPONSES_ENDPOINT,
          completion_window="24h",
        )
        return batch.id
      except openai.RateLimitError as exc:
        self._pool.mark_rate_limited(key, _parse_retry_after(exc))
      except openai.BadRequestError as exc:
        # /v1/responses not supported in batch → fall back
        print(f"[OpenAI] Batch endpoint unsupported ({exc}); switching to async mode.")
        self._use_batch = False
        raise
      except (openai.APIStatusError, openai.APIConnectionError) as exc:
        status = getattr(exc, "status_code", None)
        if status in (503,):
          self._pool.mark_rate_limited(key, 30)
        else:
          raise

  def _poll_batch(self, batch_id: str) -> dict[int, Any]:
    while True:
      key = self._pool.get_active_key()
      client = self._get_client(key)
      deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
      try:
        while time.monotonic() < deadline:
          batch = client.batches.retrieve(batch_id)
          if batch.status in ("completed", "failed", "cancelled", "expired"):
            break
          time.sleep(POLL_INTERVAL_SECONDS)
        else:
          raise TimeoutError(f"OpenAI batch {batch_id} timed out")
        break
      except openai.RateLimitError as exc:
        self._pool.mark_rate_limited(key, _parse_retry_after(exc))

    if not batch.output_file_id:
      return {}

    output_text = client.files.content(batch.output_file_id).text
    # Clean up
    try:
      client.files.delete(batch.output_file_id)
    except Exception:  # noqa: BLE001
      pass

    results: dict[int, Any] = {}
    for line in output_text.strip().splitlines():
      try:
        item = json.loads(line)
      except json.JSONDecodeError:
        continue
      idx = int(item["custom_id"])
      try:
        # Responses API wraps the response object in item["response"]["body"]
        resp_body = item["response"]["body"]
        results[idx] = _ResponseProxy(resp_body)
      except (KeyError, TypeError):
        results[idx] = None
    return results

  # ------------------------------------------------------------------ #
  # Async individual fallback                                            #
  # ------------------------------------------------------------------ #

  async def _call_one_async(
    self,
    q: Question,
    semaphore: asyncio.Semaphore,
  ) -> tuple[int, Any]:
    async with semaphore:
      while True:
        key = self._pool.get_active_key()
        async_client = self._get_async_client(key)
        try:
          body = self._build_request_body(q)
          response = await async_client.responses.create(**body)
          return q.index, response
        except openai.RateLimitError as exc:
          self._pool.mark_rate_limited(key, _parse_retry_after(exc))
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
          status = getattr(exc, "status_code", None)
          if status in (503,):
            self._pool.mark_rate_limited(key, 30)
          else:
            return q.index, None

  def _process_batch_async(self, questions: list[Question]) -> dict[int, Any]:
    semaphore = asyncio.Semaphore(_ASYNC_CONCURRENCY)

    async def _run():
      tasks = [self._call_one_async(q, semaphore) for q in questions]
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
  # BaseRunner interface                                                  #
  # ------------------------------------------------------------------ #

  def _process_batch(self, questions: list[Question]) -> dict[int, Any]:
    self._ensure_uploads(questions)

    if self._use_batch:
      try:
        pending = self._load_pending_batch()
        if pending is not None:
          saved_id, saved_indices = pending
          if saved_indices == [q.index for q in questions]:
            print(f"[OpenAI] Resuming pending batch: {saved_id}. Polling…")
            results = self._poll_batch(saved_id)
            self._clear_pending_batch()
            return results
          else:
            print(f"[OpenAI] Stale pending batch {saved_id} ignored (different questions).")
            self._clear_pending_batch()

        batch_id = self._submit_batch(questions)
        self._save_pending_batch(batch_id, questions)
        print(f"[OpenAI] Batch submitted: {batch_id}. Polling…")
        results = self._poll_batch(batch_id)
        self._clear_pending_batch()
        return results
      except openai.BadRequestError:
        self._clear_pending_batch()

    return self._process_batch_async(questions)

  def resume_from_file(self, jsonl_path: str, questions: list[Question], retry_errors: bool = False) -> None:
    """
    Parse a locally-downloaded batch output JSONL file and merge into checkpoint.

    Usage:
      uv run main.py --provider openai --checkpoint run-01 --resume-from-file /path/to/batch_output.jsonl
    """
    import checkpointing
    from result_parser import extract_result

    path = Path(jsonl_path)
    if not path.exists():
      raise FileNotFoundError(f"JSONL file not found: {path}")

    print(f"[OpenAI] Parsing batch output file: {path}")
    raw_responses: dict[int, Any] = {}
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        line = line.strip()
        if not line:
          continue
        try:
          item = json.loads(line)
        except json.JSONDecodeError:
          continue
        idx = int(item["custom_id"])
        try:
          resp_body = item["response"]["body"]
          raw_responses[idx] = _ResponseProxy(resp_body)
        except (KeyError, TypeError):
          raw_responses[idx] = None

    print(f"[OpenAI] Loaded {len(raw_responses)} responses from file.")
    self._clear_pending_batch()

    print("[OpenAI] Loading checkpoint…")
    existing = checkpointing.load_checkpoint(self.checkpoint_name, self.model_id)
    done_indices = checkpointing.answered_indices(self.checkpoint_name, self.model_id, retry_errors=retry_errors)

    if retry_errors:
      new_answers = [r for r in existing if not r.get("result", "").startswith("ERROR:")]
    else:
      new_answers = list(existing)

    print(f"[OpenAI] Parsing {len(raw_responses)} responses…")
    recovered = 0
    for q in questions:
      if q.index in done_indices or q.index not in raw_responses:
        continue
      raw = raw_responses[q.index]
      result_str, raw_output = extract_result(raw, self.provider) if raw else ("ERROR:no_response", "")
      new_answers.append({
        "index": q.index,
        "db_id": q.db_id,
        "sql_complexity": q.sql_complexity,
        "question_style": q.question_style,
        "question": q.question,
        "external_knowledge": q.external_knowledge,
        "cot": q.cot,
        "sql": q.sql,
        "result": result_str,
        "raw_output": raw_output,
      })
      recovered += 1

    print("[OpenAI] Saving checkpoint…")
    checkpointing.save_checkpoint(self.checkpoint_name, self.model_id, new_answers)
    errors = sum(1 for a in new_answers if isinstance(a, dict) and a.get("result", "").startswith("ERROR:"))
    print(f"[OpenAI] Recovered {recovered} answers → output/{self.checkpoint_name}/{self.model_id}.json  ({errors} errors)")

  def resume_batch(self, batch_id: str, questions: list[Question], retry_errors: bool = False) -> None:
    """
    Retrieve results from an already-submitted batch and merge into checkpoint.
    Use when the process was killed after submission but before results were saved.

    Usage:
      uv run main.py --provider openai --checkpoint run-01 --resume-batch batch_01...
      uv run main.py --provider openai --checkpoint run-01 --resume-batch batch_01... --retry-errors
    """
    import checkpointing
    from result_parser import extract_result

    print(f"[OpenAI] Retrieving results for batch {batch_id}…")
    raw_responses = self._poll_batch(batch_id)
    self._clear_pending_batch()

    existing = checkpointing.load_checkpoint(self.checkpoint_name, self.model_id)
    done_indices = checkpointing.answered_indices(self.checkpoint_name, self.model_id, retry_errors=retry_errors)

    if retry_errors:
      new_answers = [r for r in existing if not r.get("result", "").startswith("ERROR:")]
    else:
      new_answers = list(existing)

    recovered = 0
    for q in questions:
      if q.index in done_indices or q.index not in raw_responses:
        continue
      raw = raw_responses[q.index]
      result_str, raw_output = extract_result(raw, self.provider) if raw else ("ERROR:no_response", "")
      new_answers.append({
        "index": q.index,
        "db_id": q.db_id,
        "sql_complexity": q.sql_complexity,
        "question_style": q.question_style,
        "question": q.question,
        "external_knowledge": q.external_knowledge,
        "cot": q.cot,
        "sql": q.sql,
        "result": result_str,
        "raw_output": raw_output,
      })
      recovered += 1

    checkpointing.save_checkpoint(self.checkpoint_name, self.model_id, new_answers)
    print(f"[OpenAI] Recovered {recovered} answers → output/{self.checkpoint_name}/{self.model_id}.json")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

class _ResponseProxy:
  """Wrap a raw Responses API JSON body (from batch output) to mimic the
  live SDK response object expected by result_parser.extract_result_openai."""

  def __init__(self, body: dict):
    self._body = body

  @property
  def output(self):
    return [_OutputItemProxy(item) for item in self._body.get("output") or []]


class _OutputItemProxy:
  def __init__(self, item: dict):
    self._item = item

  @property
  def type(self):
    return self._item.get("type")

  @property
  def outputs(self):
    return [_OutputProxy(o) for o in self._item.get("outputs") or []]

  @property
  def content(self):
    return [_ContentProxy(c) for c in self._item.get("content") or []]


class _OutputProxy:
  def __init__(self, o: dict):
    self._o = o

  @property
  def type(self):
    return self._o.get("type")

  @property
  def logs(self):
    return self._o.get("logs", "")


class _ContentProxy:
  def __init__(self, c: dict):
    self._c = c

  @property
  def type(self):
    return self._c.get("type")

  @property
  def text(self):
    return self._c.get("text", "")


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
