"""
Claude runner — Batch Messages API + Files API + code_execution tool.

Strategy:
1. Upload each CSV once via Files API (beta "files-api-2025-04-14"); cache file_ids.
   If the Files API is unavailable (404/403), falls back to embedding CSV inline.
2. Build batch requests with container_upload (or inline) + code_execution_20250825.
   The CSV context block carries cache_control=ephemeral for prompt caching.
3. Submit via client.messages.batches.create(); poll until ended.
4. Fall back to async concurrent individual calls on BadRequestError.

Docs reference:
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from api_key_pool import ApiKeyPool
from config import CSV_REGISTRY, OUTPUT_DIR, POLL_INTERVAL_SECONDS, POLL_TIMEOUT_SECONDS
from prompt import SYSTEM_PROMPT, build_user_prompt
from runners.base import BaseRunner, Question

# Files API beta — required for client.beta.files.upload + container_upload blocks
_BETA_FILES = "files-api-2025-04-14"
_CODE_EXEC_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
_MAX_TOKENS = 4096
_ASYNC_CONCURRENCY = 5
_FILE_ID_CACHE_PATH: Path = OUTPUT_DIR / ".file_id_cache.json"


class ClaudeRunner(BaseRunner):
  provider = "claude"

  def __init__(self, model_id: str, checkpoint_name: str, max_rows: int | None = None):
    super().__init__(model_id, checkpoint_name, max_rows)
    self._pool = ApiKeyPool("anthropic")
    self._file_id_cache: dict[str, str] = self._load_file_id_cache()
    # When max_rows is set we can't use the Files API (it would upload the full CSV).
    self._files_api_available: bool | None = False if max_rows is not None else None
    self._csv_cache: dict[str, str] = {}
    # Batch API saves 50% on cost but has up to 24 h turnaround; disabled by default
    # so interactive runs use the faster async path. Set True to opt in.
    self._use_batch = False

  @staticmethod
  def _load_file_id_cache() -> dict[str, str]:
    if _FILE_ID_CACHE_PATH.exists():
      try:
        return json.loads(_FILE_ID_CACHE_PATH.read_text(encoding="utf-8"))
      except (json.JSONDecodeError, OSError):
        pass
    return {}

  @staticmethod
  def _save_file_id_cache(cache: dict[str, str]) -> None:
    _FILE_ID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FILE_ID_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

  def _get_client(self, key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=key)

  # ------------------------------------------------------------------ #
  # File upload (Files API)                                              #
  # ------------------------------------------------------------------ #

  def _try_upload_csv(self, db_id: str) -> str | None:
    """
    Upload CSV via Files API. Returns file_id on success, None if the
    Files API is unavailable for this account (404/403).
    Blocks and retries indefinitely on rate limits.
    """
    if db_id in self._file_id_cache:
      return self._file_id_cache[db_id]

    meta = CSV_REGISTRY[db_id]
    key = self._pool.get_active_key()
    client = self._get_client(key)

    while True:
      try:
        with open(meta.path, "rb") as fh:
          file_obj = client.beta.files.upload(
            file=fh
            # file=(meta.filename, fh, "text/csv"),
          )
        self._file_id_cache[db_id] = file_obj.id
        self._save_file_id_cache(self._file_id_cache)
        print(f"[Claude] Uploaded {meta.filename} → {file_obj.id}")
        return file_obj.id
      except (anthropic.NotFoundError, anthropic.PermissionDeniedError) as exc:
        print(f"[Claude] Files API unavailable ({exc.status_code}); using inline CSV.")
        return None
      except anthropic.RateLimitError as exc:
        retry_after = _parse_retry_after(exc)
        self._pool.mark_rate_limited(key, retry_after)
        key = self._pool.get_active_key()
        client = self._get_client(key)
      except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        status = getattr(exc, "status_code", None)
        if status in (503, 529):
          self._pool.mark_rate_limited(key, 30)
          key = self._pool.get_active_key()
          client = self._get_client(key)
        else:
          raise

  def _ensure_uploads(self, questions: list[Question]) -> None:
    """
    Try to upload all CSVs for this batch via the Files API.
    On first 404/403, marks files_api_available=False and skips the rest.
    """
    if self._files_api_available is False:
      return  # already know it's unavailable

    for db_id in {q.db_id for q in questions}:
      if db_id in self._file_id_cache:
        continue
      file_id = self._try_upload_csv(db_id)
      if file_id is None:
        self._files_api_available = False
        return  # don't try other files
    self._files_api_available = True

  # ------------------------------------------------------------------ #
  # Request building                                                     #
  # ------------------------------------------------------------------ #

  def _get_csv_content(self, db_id: str) -> str:
    if db_id not in self._csv_cache:
      meta = CSV_REGISTRY[db_id]
      text = meta.path.read_text(encoding=meta.encoding)
      if self.max_rows is not None:
        text = self._trim_csv(text, self.max_rows)
      self._csv_cache[db_id] = text
    return self._csv_cache[db_id]

  def _build_messages(self, q: Question) -> list[dict]:
    meta = CSV_REGISTRY[q.db_id]
    file_id = self._file_id_cache.get(q.db_id)

    if file_id:
      # Files API path: container_upload block carries the CSV into the sandbox.
      # The text block with metadata+question is NOT cached (changes per question).
      return [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": build_user_prompt(q.question, q.db_id, q.external_knowledge, meta),
            },
            {
              "type": "container_upload",
              "file_id": file_id,
            },
          ],
        }
      ]
    else:
      # Inline fallback: embed full CSV in the message with cache_control so that
      # questions sharing the same db_id get prompt-cache hits.
      columns_str = ", ".join(meta.columns)
      rows_note = (
        f"NOTE: Only the first {self.max_rows} rows of this CSV are provided. "
        f"The full dataset may contain more rows.\n"
        if self.max_rows is not None
        else ""
      )
      csv_context = (
        f"DATASET: {q.db_id}\n"
        f"DELIMITER: {meta.delimiter!r}  |  ENCODING: {meta.encoding!r}\n"
        f"COLUMNS: {columns_str}\n"
        f"{rows_note}"
        f"\nCSV DATA:\n{self._get_csv_content(q.db_id)}"
      )
      knowledge_section = (
        f"\nEXTERNAL KNOWLEDGE (important — use this to interpret the question correctly):\n"
        f"{q.external_knowledge.strip()}\n"
        if q.external_knowledge and q.external_knowledge.strip()
        else ""
      )
      question_text = f"{knowledge_section}\nQUESTION: {q.question}"
      return [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": csv_context,
              "cache_control": {"type": "ephemeral"},
            },
            {
              "type": "text",
              "text": question_text,
            },
          ],
        }
      ]

  def _build_system_block(self) -> list[dict]:
    return [
      {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
      }
    ]

  def _request_betas(self) -> list[str]:
    """Beta headers to include per-request (only needed for Files API path)."""
    return [_BETA_FILES] if self._files_api_available else []

  # ------------------------------------------------------------------ #
  # Batch API — pending batch persistence                               #
  # ------------------------------------------------------------------ #

  def _pending_batch_path(self) -> Path:
    from config import OUTPUT_DIR
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
    path = self._pending_batch_path()
    path.unlink(missing_ok=True)

  # ------------------------------------------------------------------ #
  # Batch API                                                            #
  # ------------------------------------------------------------------ #

  def _submit_batch(self, questions: list[Question]) -> str:
    key = self._pool.get_active_key()
    client = self._get_client(key)
    betas = self._request_betas()

    requests = [
      Request(
        custom_id=str(q.index),
        params=MessageCreateParamsNonStreaming(
          model=self.model_id,
          max_tokens=_MAX_TOKENS,
          system=self._build_system_block(),
          messages=self._build_messages(q),
          tools=[_CODE_EXEC_TOOL],
        ),
      )
      for q in questions
    ]

    while True:
      try:
        if betas:
          batch = client.beta.messages.batches.create(requests=requests, betas=betas)
        else:
          batch = client.messages.batches.create(requests=requests)
        return batch.id
      except anthropic.RateLimitError as exc:
        self._pool.mark_rate_limited(key, _parse_retry_after(exc))
        key = self._pool.get_active_key()
        client = self._get_client(key)
      except anthropic.BadRequestError as exc:
        print(f"[Claude] Batch API rejected ({exc}); switching to async mode.")
        self._use_batch = False
        raise
      except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        status = getattr(exc, "status_code", None)
        if status in (503, 529):
          self._pool.mark_rate_limited(key, 30)
          key = self._pool.get_active_key()
          client = self._get_client(key)
        else:
          raise

  def _poll_batch(self, batch_id: str) -> dict[int, Any]:
    key = self._pool.get_active_key()
    client = self._get_client(key)
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
      try:
        status = client.messages.batches.retrieve(batch_id)
        if status.processing_status == "ended":
          break
        time.sleep(POLL_INTERVAL_SECONDS)
      except anthropic.RateLimitError as exc:
        self._pool.mark_rate_limited(key, _parse_retry_after(exc))
        key = self._pool.get_active_key()
        client = self._get_client(key)
    else:
      raise TimeoutError(f"Batch {batch_id} did not complete within timeout")

    results: dict[int, Any] = {}
    for result in client.messages.batches.results(batch_id):
      idx = int(result.custom_id)
      if result.result.type == "succeeded":
        results[idx] = result.result.message
      else:
        results[idx] = None
    return results

  # ------------------------------------------------------------------ #
  # Async individual fallback                                            #
  # ------------------------------------------------------------------ #

  async def _call_one_async(
    self,
    q: Question,
    semaphore: asyncio.Semaphore,
    clients: dict[str, anthropic.AsyncAnthropic],
  ) -> tuple[int, Any]:
    async with semaphore:
      betas = self._request_betas()
      messages = self._build_messages(q)
      last_response = None
      # Bound prevents an unbounded loop if the model emits pause_turn repeatedly.
      max_continuations = 5

      for _ in range(max_continuations):
        while True:
          key = await self._pool.get_active_key_async()
          async_client = clients[key]
          try:
            if betas:
              response = await async_client.beta.messages.create(
                model=self.model_id,
                max_tokens=_MAX_TOKENS,
                system=self._build_system_block(),
                messages=messages,
                tools=[_CODE_EXEC_TOOL],
                betas=betas,
              )
            else:
              response = await async_client.messages.create(
                model=self.model_id,
                max_tokens=_MAX_TOKENS,
                system=self._build_system_block(),
                messages=messages,
                tools=[_CODE_EXEC_TOOL],
              )
            last_response = response
            break  # successful call
          except anthropic.RateLimitError as exc:
            self._pool.mark_rate_limited(key, _parse_retry_after(exc))
          except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            if status in (503, 529):
              self._pool.mark_rate_limited(key, 30)
            else:
              return q.index, None

        # If pause_turn, continue with the conversation so the server resumes
        if getattr(last_response, "stop_reason", None) == "pause_turn":
          messages = self._build_messages(q) + [
            {"role": "assistant", "content": last_response.content},
          ]
        else:
          break  # end_turn or other terminal stop reason

      return q.index, last_response

  def _process_batch_async(self, questions: list[Question]) -> dict[int, Any]:
    semaphore = asyncio.Semaphore(_ASYNC_CONCURRENCY)

    async def _run():
      clients = {key: anthropic.AsyncAnthropic(api_key=key) for key in self._pool._keys}
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
        idx, resp = outcome
        results[idx] = resp
    return results

  def _batch_size(self) -> int:
    # Batch API: large chunks are fine (submitted at once, polled later).
    # Async mode: small chunks so checkpoints are saved frequently and
    # progress isn't lost on rate-limit sleeps or interruptions.
    return 10000 if self._use_batch else _ASYNC_CONCURRENCY

  # ------------------------------------------------------------------ #
  # BaseRunner interface                                                 #
  # ------------------------------------------------------------------ #

  def _process_batch(self, questions: list[Question]) -> dict[int, Any]:
    # Try Files API first; sets _files_api_available after first probe
    self._ensure_uploads(questions)

    if self._use_batch:
      try:
        # Resume a pending batch if it covers the same questions
        pending = self._load_pending_batch()
        if pending is not None:
          saved_id, saved_indices = pending
          if saved_indices == [q.index for q in questions]:
            print(f"[Claude] Resuming pending batch: {saved_id}. Polling…")
            results = self._poll_batch(saved_id)
            self._clear_pending_batch()
            return results
          else:
            print(f"[Claude] Stale pending batch {saved_id} ignored (different questions).")
            self._clear_pending_batch()

        batch_id = self._submit_batch(questions)
        self._save_pending_batch(batch_id, questions)
        mode = "files" if self._files_api_available else "inline"
        print(f"[Claude] Batch submitted ({mode}): {batch_id}. Polling…")
        results = self._poll_batch(batch_id)
        self._clear_pending_batch()
        return results
      except anthropic.BadRequestError:
        self._clear_pending_batch()  # _use_batch already set to False

    return self._process_batch_async(questions)

  def resume_batch(self, batch_id: str, questions: list[Question], retry_errors: bool = False) -> None:
    """
    Retrieve results from an already-submitted batch and merge into checkpoint.
    Use when the process was killed after submission but before results were saved.

    Usage:
      uv run main.py --provider claude --checkpoint run-01 --resume-batch msgbatch_01...
      uv run main.py --provider claude --checkpoint run-01 --resume-batch msgbatch_01... --retry-errors
    """
    import checkpointing
    from result_parser import extract_result

    print(f"[Claude] Retrieving results for batch {batch_id}…")
    raw_responses = self._poll_batch(batch_id)
    self._clear_pending_batch()

    existing = checkpointing.load_checkpoint(self.checkpoint_name, self.model_id)
    done_indices = checkpointing.answered_indices(self.checkpoint_name, self.model_id, retry_errors=retry_errors)

    # Drop error entries that should be retried
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
    print(f"[Claude] Recovered {recovered} answers → output/{self.checkpoint_name}/{self.model_id}.json")


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

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
