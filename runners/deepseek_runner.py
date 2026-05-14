"""
DeepSeek runner — OpenAI-compatible Chat Completions API with tool-calling loop.

Unlike claude/gemini/openai runners, DeepSeek has no hosted code execution.
We declare a `run_python` tool and execute it locally via subprocess, then feed
the stdout back into the conversation so the model can iterate until it has the
final answer.

Design:
1. Async concurrency (DEEPSEEK_CONCURRENCY = 20) with asyncio.Semaphore.
2. Per-question checkpoint: run() is overridden to save after every completed question
   using asyncio.as_completed(), so a crash never loses more than in-flight work.
3. Multi-step tool loop: model calls run_python up to _MAX_TOOL_ROUNDS times.
   On error, the stderr is returned so the model can self-correct.
4. API key rotation using ApiKeyPool on rate-limit / transient failures.
5. reasoning_content fix: DeepSeek v4 models return a `reasoning_content` field in
   thinking mode; the API requires it to be passed back in subsequent turns.
   We use msg.model_dump() to capture all fields including reasoning_content.

Debug mode:
  Set DEEPSEEK_DEBUG=1 in .env to save each question's full conversation +
  execution trace to output/deepseek_debug/<checkpoint>/<model_id>/q<idx>.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import openai
from openai import AsyncOpenAI
from tqdm import tqdm

import checkpointing
from api_key_pool import ApiKeyPool
from config import CSV_REGISTRY, DEEPSEEK_CONCURRENCY, OUTPUT_DIR
from evaluator import is_column_order_mismatch_result
from prompt import DEEPSEEK_SYSTEM_SUFFIX, SYSTEM_PROMPT, build_question_prompt, build_static_context
from result_parser import extract_result
from runners.base import AnsweredQuestion, BaseRunner, Question

_MAX_TOKENS = 8192
_MAX_TOOL_ROUNDS = 8   # early exit handles simple cases; complex exploration needs room
_REQUEST_TIMEOUT_SECONDS = 180
_LOCAL_EXEC_TIMEOUT_SECONDS = 40
_DEFAULT_BASE_URL = "https://api.deepseek.com"

# Fields to keep from msg.model_dump() when building the assistant turn.
# When thinking is enabled, DeepSeek v4 requires reasoning_content echoed back.
# When thinking is disabled (default), reasoning_content is absent / empty.
# All other OpenAI SDK fields (refusal, audio, etc.) are omitted.
_ASSISTANT_FIELDS = {"role", "content", "tool_calls", "reasoning_content"}

_RUN_PYTHON_TOOL: dict = {
  "type": "function",
  "function": {
    "name": "run_python",
    "description": (
      "Execute a Python script on the CSV dataset for exploration or computation. "
      "The CSV file is available as 'data.csv' in the working directory. "
      "Returns stdout on success, or an error message on failure. "
      "Use this tool to explore data; use submit_answer to submit the final result."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Complete Python script to execute using pandas.",
        }
      },
      "required": ["code"],
      "additionalProperties": False,
    },
  },
}

_SUBMIT_ANSWER_TOOL: dict = {
  "type": "function",
  "function": {
    "name": "submit_answer",
    "description": (
      "Submit the final answer. Call this exactly once when you have the definitive result. "
      "result must be an array of arrays — each inner array is one result row."
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "result": {
          "type": "array",
          "description": (
            "Array of result rows. Each row is an array of values. "
            "E.g. [[\"Alice\", 30], [\"Bob\", 25]]. Empty result: []. Single scalar: [[42]]."
          ),
          "items": {"type": "array", "items": {}},
        }
      },
      "required": ["result"],
      "additionalProperties": False,
    },
  },
}


def _is_debug() -> bool:
  return os.getenv("DEEPSEEK_DEBUG", "").strip() in ("1", "true", "yes")


def _thinking_extra_body() -> dict | None:
  """
  Return extra_body for thinking mode.

  DEEPSEEK_THINKING=low (default) → reasoning_effort=low
    Keeps light chain-of-thought for code correctness, cuts ~60-70% of
    reasoning tokens vs the model default ("max" / "high").

  DEEPSEEK_THINKING=disabled → {"thinking": {"type": "disabled"}}
    Zero reasoning tokens — cheapest, but may hurt accuracy on complex queries.

  DEEPSEEK_THINKING=enabled → None (use the model's default reasoning budget)
    Full thinking — most accurate, most expensive.
  """
  mode = os.getenv("DEEPSEEK_THINKING", "low").strip().lower()
  if mode == "disabled":
    return {"thinking": {"type": "disabled"}}
  if mode == "enabled":
    return None
  # default: low
  return {"reasoning_effort": "low"}


def _save_debug(
  checkpoint: str,
  model_id: str,
  q: Question,
  messages: list[dict],
  code_parts: list[str],
  exec_outputs: list[str],
  final_text: str,
  submitted_answer: list | None,
  tool_limit_exceeded: bool,
) -> None:
  debug_dir = OUTPUT_DIR / "deepseek_debug" / checkpoint / model_id.replace("/", "_")
  debug_dir.mkdir(parents=True, exist_ok=True)
  payload = {
    "question_index": q.index,
    "db_id": q.db_id,
    "question": q.question,
    "messages": messages,
    "code_parts": code_parts,
    "exec_outputs": exec_outputs,
    "final_text": final_text,
    "submitted_answer": submitted_answer,
    "tool_limit_exceeded": tool_limit_exceeded,
  }
  out = debug_dir / f"q{q.index}.json"
  out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DeepSeekRunner(BaseRunner):
  provider = "deepseek"

  def __init__(self, model_id: str, checkpoint_name: str, max_rows: int | None = None):
    super().__init__(model_id, checkpoint_name, max_rows)
    self._pool = ApiKeyPool("deepseek")
    self._csv_cache: dict[str, str] = {}
    self._csv_cache_full: dict[str, str] = {}
    self._debug = _is_debug()

  def _build_client(self, key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
      api_key=key,
      base_url=os.getenv("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/"),
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

  def _build_messages(self, q: Question) -> list[dict]:
    meta = CSV_REGISTRY[q.db_id]
    csv_text = self._csv_text(q.db_id)
    # All static content (CSV + dataset metadata) goes in system prompt so every
    # question sharing the same db_id has an identical prefix → DeepSeek's
    # automatic KV cache hits from question 2 onward.
    system_text = (
      SYSTEM_PROMPT
      + DEEPSEEK_SYSTEM_SUFFIX
      + f"\n\n<csv filename=\"{meta.filename}\">\n{csv_text}\n</csv>"
      + "\n\n"
      + build_static_context(q.db_id, meta, self.max_rows)
    )
    user_text = build_question_prompt(q.question, q.external_knowledge)
    return [
      {"role": "system", "content": system_text},
      {"role": "user", "content": user_text},
    ]

  def _execute_python_code(self, code: str, q: Question) -> str:
    """
    Run `code` in a temp dir alongside the CSV. Returns stdout on success or
    a short error string on failure (so the model can self-correct).
    """
    meta = CSV_REGISTRY[q.db_id]
    csv_text = self._csv_text_full(q.db_id)
    with tempfile.TemporaryDirectory(prefix="deepseek_exec_") as tmpdir:
      script_path = os.path.join(tmpdir, "script.py")
      data_path = os.path.join(tmpdir, "data.csv")
      alt_path = os.path.join(tmpdir, meta.filename)
      try:
        with open(script_path, "w", encoding="utf-8") as fh:
          fh.write(code)
        with open(data_path, "w", encoding="utf-8") as fh:
          fh.write(csv_text)
        with open(alt_path, "w", encoding="utf-8") as fh:
          fh.write(csv_text)
      except OSError as exc:
        return f"OSError writing temp files: {exc}"

      try:
        proc = subprocess.run(
          [sys.executable, script_path],
          cwd=tmpdir,
          capture_output=True,
          text=True,
          timeout=_LOCAL_EXEC_TIMEOUT_SECONDS,
          check=False,
        )
      except subprocess.TimeoutExpired:
        return "Error: execution timed out"
      except (subprocess.SubprocessError, OSError) as exc:
        return f"Error: {exc}"

      if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        return f"Error (exit {proc.returncode}):\n{stderr[:1000]}"

      stdout = (proc.stdout or "").strip()
      return stdout if stdout else "(no output)"

  @staticmethod
  def _build_assistant_msg(msg: Any) -> dict:
    """
    Serialize the assistant message for the next turn.
    Uses model_dump() to capture DeepSeek-specific fields (reasoning_content)
    that must be echoed back to the API when thinking mode is active.
    """
    raw = msg.model_dump()
    result: dict = {}
    for field in _ASSISTANT_FIELDS:
      if field in raw and raw[field] is not None:
        result[field] = raw[field]
    # role and content are always required even if empty/None
    result.setdefault("role", "assistant")
    result.setdefault("content", raw.get("content") or "")
    return result

  async def _call_one_async(
    self,
    q: Question,
    semaphore: asyncio.Semaphore,
    clients: dict[str, AsyncOpenAI],
  ) -> tuple[int, Any]:
    async with semaphore:
      messages = self._build_messages(q)
      code_parts: list[str] = []
      exec_outputs: list[str] = []
      final_text = ""
      submitted_answer: list | None = None
      tool_limit_exceeded = False

      for _round in range(_MAX_TOOL_ROUNDS):
        response = None
        extra_body = _thinking_extra_body()
        for _ in range(8):
          key = await self._pool.get_active_key_async()
          client = clients[key]
          try:
            response = await asyncio.wait_for(
              client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=[_RUN_PYTHON_TOOL, _SUBMIT_ANSWER_TOOL],
                tool_choice="auto",
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                **({"extra_body": extra_body} if extra_body else {}),
              ),
              timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            break  # success
          except asyncio.TimeoutError:
            self._pool.mark_rate_limited(key, 15)
            continue
          except openai.RateLimitError as exc:
            self._pool.mark_rate_limited(key, _parse_retry_after(exc))
            continue
          except (openai.APIStatusError, openai.APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            print(
              f"\n[deepseek q{q.index}] APIError status={status}: {exc}",
              flush=True,
            )
            if _is_retryable_status(status):
              self._pool.mark_rate_limited(key, 30)
              continue
            return q.index, None
          except Exception as exc:  # noqa: BLE001
            err = str(exc).lower()
            print(f"\n[deepseek q{q.index}] Exception: {type(exc).__name__}: {exc}", flush=True)
            if any(t in err for t in ("rate limit", "temporar", "timeout", "overloaded", "busy")):
              self._pool.mark_rate_limited(key, 20)
              continue
            return q.index, None
        else:
          print(f"\n[deepseek q{q.index}] All attempts exhausted at round {_round}", flush=True)
          return q.index, None

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message

        # Build assistant message, preserving reasoning_content for DeepSeek thinking mode
        assistant_msg = self._build_assistant_msg(msg)
        messages.append(assistant_msg)

        if finish_reason == "tool_calls" and msg.tool_calls:
          submit_called = False
          for tc in msg.tool_calls:
            if tc.function.name == "run_python":
              try:
                args = json.loads(tc.function.arguments)
                code = args.get("code", "")
              except (json.JSONDecodeError, KeyError):
                code = ""
              if code:
                code_parts.append(code)
                stdout = self._execute_python_code(code, q)
                exec_outputs.append(stdout)
                messages.append({
                  "role": "tool",
                  "tool_call_id": tc.id,
                  "content": stdout,
                })
              else:
                messages.append({
                  "role": "tool",
                  "tool_call_id": tc.id,
                  "content": "Error: empty code.",
                })
            elif tc.function.name == "submit_answer":
              try:
                args = json.loads(tc.function.arguments)
                submitted_answer = args.get("result")
              except (json.JSONDecodeError, KeyError):
                submitted_answer = None
              messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Answer submitted.",
              })
              submit_called = True
              break  # stop processing further tool calls in this batch
            else:
              messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Unknown tool: {tc.function.name}",
              })

          if submit_called:
            break  # exit round loop — we have the answer

          # Convergence guard: two identical run_python outputs → stuck loop.
          if len(exec_outputs) >= 2 and exec_outputs[-1] == exec_outputs[-2]:
            break
        else:
          # Model stopped calling tools — capture final text and exit loop
          final_text = msg.content or ""
          break
      else:
        # for loop completed all _MAX_TOOL_ROUNDS without a break
        tool_limit_exceeded = True

      if self._debug:
        _save_debug(
          self.checkpoint_name,
          self.model_id,
          q,
          messages,
          code_parts,
          exec_outputs,
          final_text,
          submitted_answer,
          tool_limit_exceeded,
        )

      return q.index, SimpleNamespace(
        code_parts=code_parts,
        exec_outputs=exec_outputs,
        final_text=final_text,
        submitted_answer=submitted_answer,
        tool_limit_exceeded=tool_limit_exceeded,
      )

  # _process_batch is abstract in BaseRunner — stub to satisfy ABC.
  def _process_batch(self, _questions: list[Question]) -> dict[int, Any]:
    raise NotImplementedError("DeepSeekRunner uses run() directly")

  # ------------------------------------------------------------------
  # Override run() for per-question checkpointing with full concurrency
  # ------------------------------------------------------------------

  def run(
    self,
    questions: list[Question],
    retry_errors: bool = False,
  ) -> list[AnsweredQuestion]:
    done = checkpointing.answered_indices(
      self.checkpoint_name, self.model_id, retry_errors=retry_errors
    )
    existing = checkpointing.load_checkpoint(self.checkpoint_name, self.model_id)

    remaining = sorted(
      [q for q in questions if q.index not in done],
      key=lambda q: q.db_id,
    )

    if not remaining:
      print(f"[{self.model_id}] All {len(questions)} questions already answered.")
      return [
        AnsweredQuestion(**{**{"raw_output": ""}, **r}) if isinstance(r, dict) else r
        for r in existing
      ]

    print(
      f"[{self.model_id}] {len(done)} already done, "
      f"{len(remaining)} remaining. Concurrency: {DEEPSEEK_CONCURRENCY}"
    )
    if self._debug:
      debug_dir = OUTPUT_DIR / "deepseek_debug" / self.checkpoint_name
      print(f"[{self.model_id}] Debug mode ON → {debug_dir}")

    if retry_errors:
      base_results: list = [
        r for r in existing
        if not (isinstance(r, dict) and r.get("result", "").startswith("ERROR:"))
      ]
    else:
      base_results = list(existing)

    remaining_by_index = {q.index: q for q in remaining}

    async def _run_async() -> list:
      semaphore = asyncio.Semaphore(DEEPSEEK_CONCURRENCY)
      clients = {key: self._build_client(key) for key in self._pool._keys}
      results = list(base_results)

      try:
        tasks = [
          asyncio.create_task(self._call_one_async(q, semaphore, clients))
          for q in remaining
        ]
        with tqdm(total=len(remaining), desc=self.model_id, unit="q") as pbar:
          for coro in asyncio.as_completed(tasks):
            idx, raw = await coro
            q = remaining_by_index[idx]

            if raw is None:
              result_str, raw_output = "ERROR:no_response", ""
            else:
              result_str, raw_output = extract_result(raw, self.provider)

            # Auto-retry once for column-order mistakes
            if raw is not None and not result_str.startswith("ERROR:"):
              if is_column_order_mismatch_result(q, result_str):
                retry_q = Question(
                  index=q.index,
                  db_id=q.db_id,
                  sql_complexity=q.sql_complexity,
                  question_style=q.question_style,
                  question=q.question,
                  external_knowledge=(
                    (q.external_knowledge + "\n\n") if q.external_knowledge else ""
                  ) + self._column_order_retry_hint(),
                  cot=q.cot,
                  sql=q.sql,
                )
                _, retry_raw = await self._call_one_async(retry_q, semaphore, clients)
                if retry_raw is not None:
                  retry_result, retry_raw_output = extract_result(retry_raw, self.provider)
                  if not retry_result.startswith("ERROR:"):
                    result_str, raw_output = retry_result, retry_raw_output

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

            # Save after every completed question
            checkpointing.save_checkpoint(self.checkpoint_name, self.model_id, results)

      finally:
        for client in clients.values():
          await client.close()

      return results

    results = asyncio.run(_run_async())
    return [
      AnsweredQuestion(**{**{"raw_output": ""}, **r}) if isinstance(r, dict) else r
      for r in results
    ]


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
