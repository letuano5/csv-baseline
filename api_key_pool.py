"""
API key pool with automatic rotation on rate-limit / quota / server errors.

Usage:
  pool = ApiKeyPool("anthropic")
  key = pool.get_active_key()
  try:
      result = call_api(key, ...)
  except RateLimitError as e:
      pool.mark_rate_limited(key, retry_after=parse_retry_after(e))
      raise   # caller should retry

Env var patterns (any of these work):
  ANTHROPIC_API_KEY            (single key)
  ANTHROPIC_API_KEY_1, _2 ...  (multiple keys, 1-indexed)
  OPENAI_API_KEY / OPENAI_API_KEY_1 ...
  GOOGLE_API_KEY / GOOGLE_API_KEY_1 ...
  OPENROUTER_API_KEY / OPENROUTER_API_KEY_1 ...
"""

import asyncio
import os
import time
from collections import deque

_ENV_PREFIXES: dict[str, str] = {
  "anthropic": "ANTHROPIC_API_KEY",
  "openai": "OPENAI_API_KEY",
  "gemini": "GOOGLE_API_KEY",
  "openrouter": "OPENROUTER_API_KEY",
  "deepseek": "DEEPSEEK_API_KEY",
}

_SLEEP_ALL_EXHAUSTED = 360  # 6 minutes when every key is blocked
_DEFAULT_BACKOFF = 60       # seconds to block a key if no Retry-After header


class ApiKeyPool:
  def __init__(self, provider: str):
    prefix = _ENV_PREFIXES.get(provider)
    if prefix is None:
      raise ValueError(f"Unknown provider: {provider!r}. Known: {list(_ENV_PREFIXES)}")
    self._keys: deque[str] = deque(self._load_keys(prefix))
    if not self._keys:
      raise RuntimeError(
        f"No API keys found for provider {provider!r}. "
        f"Set {prefix} or {prefix}_1, {prefix}_2 ... in your .env"
      )
    self._blocked_until: dict[str, float] = {}

  @staticmethod
  def _load_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    # First try bare name (single-key convention)
    bare = os.getenv(prefix, "").strip()
    if bare:
      keys.append(bare)
    # Then try indexed keys: PREFIX_1, PREFIX_2, ...
    idx = 1
    while True:
      val = os.getenv(f"{prefix}_{idx}", "").strip()
      if not val:
        break
      if val not in keys:
        keys.append(val)
      idx += 1
    return keys

  def get_active_key(self) -> str:
    """Return the first non-blocked key. Sleeps if all keys are currently blocked."""
    while True:
      now = time.monotonic()
      for key in self._keys:
        if self._blocked_until.get(key, 0.0) <= now:
          return key
      # All keys blocked — sleep until the earliest one unblocks
      earliest_unblock = min(self._blocked_until.values())
      wait = min(_SLEEP_ALL_EXHAUSTED, max(1.0, earliest_unblock - now + 1.0))
      print(
        f"[ApiKeyPool] All {len(self._keys)} key(s) exhausted. "
        f"Sleeping {wait:.0f}s before retry…"
      )
      time.sleep(wait)

  async def get_active_key_async(self) -> str:
    """Async version — uses asyncio.sleep so the event loop stays unblocked."""
    while True:
      now = time.monotonic()
      for key in self._keys:
        if self._blocked_until.get(key, 0.0) <= now:
          return key
      earliest_unblock = min(self._blocked_until.values())
      wait = min(_SLEEP_ALL_EXHAUSTED, max(1.0, earliest_unblock - now + 1.0))
      print(
        f"[ApiKeyPool] All {len(self._keys)} key(s) exhausted. "
        f"Sleeping {wait:.0f}s before retry…"
      )
      await asyncio.sleep(wait)

  def peek_available_key(self) -> str | None:
    """Return a non-blocked key without sleeping, or None if all keys are blocked."""
    now = time.monotonic()
    for key in self._keys:
      if self._blocked_until.get(key, 0.0) <= now:
        return key
    return None

  def mark_rate_limited(self, key: str, retry_after: float | None = None) -> None:
    """Block a key for `retry_after` seconds (default: {_DEFAULT_BACKOFF}s)."""
    backoff = retry_after if retry_after is not None else _DEFAULT_BACKOFF
    self._blocked_until[key] = time.monotonic() + backoff
    # Rotate so that the next get_active_key() tries a different key first
    try:
      idx = list(self._keys).index(key)
    except ValueError:
      return
    self._keys.rotate(-idx - 1)

  def __len__(self) -> int:
    return len(self._keys)
