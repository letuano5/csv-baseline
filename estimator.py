"""
Estimate token usage and API cost before running a full job.

Usage:
  uv run main.py --estimate                      # all default models
  uv run main.py --provider claude --estimate    # single provider
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import CSV_REGISTRY, AVG_OUTPUT_TOKENS, AVG_PROMPT_TOKENS

if TYPE_CHECKING:
  from runners.base import Question


# ---------------------------------------------------------------------------
# Pricing table  (per-1M tokens unless noted)
# ---------------------------------------------------------------------------

@dataclass
class ModelPricing:
  input: float          # standard input price / 1M
  output: float         # output price / 1M
  cache_write_mult: float  # multiplier on input price for cache write
  cache_read: float     # cache read price / 1M
  batch_discount: float # fraction to multiply (0.5 = 50% off)
  # Gemini has two input tiers by context length
  input_long: float | None = None   # price for context > 200K tokens
  cache_read_long: float | None = None


PRICING: dict[str, ModelPricing] = {
  "claude-sonnet-4-6": ModelPricing(
    input=3.00, output=15.00,
    cache_write_mult=1.25, cache_read=0.30,
    batch_discount=0.50,
  ),
  "gemini-3.1-pro-preview": ModelPricing(
    input=2.00, output=12.00,
    cache_write_mult=1.00, cache_read=0.20,
    batch_discount=0.50,
    input_long=4.00, cache_read_long=0.40,
  ),
  "gpt-5.4": ModelPricing(
    input=2.50, output=15.00,
    cache_write_mult=1.00, cache_read=0.25,
    batch_discount=0.50,
  ),
}

LONG_CONTEXT_THRESHOLD = 200_000  # tokens; Gemini pricing tier boundary


# ---------------------------------------------------------------------------
# Core estimation logic
# ---------------------------------------------------------------------------

def _csv_tokens_for_db(db_id: str) -> int:
  return CSV_REGISTRY[db_id].approx_tokens


def estimate(
  questions: list["Question"],
  model_id: str,
  *,
  use_batch: bool = True,
  use_cache: bool = True,
  avg_output_tokens: int = AVG_OUTPUT_TOKENS,
  avg_prompt_tokens: int = AVG_PROMPT_TOKENS,
) -> None:
  pricing = PRICING.get(model_id)
  if pricing is None:
    print(f"[estimate] No pricing data for model {model_id!r}. Using claude-sonnet-4-6 as proxy.")
    pricing = PRICING["claude-sonnet-4-6"]

  count_by_db: Counter[str] = Counter(q.db_id for q in questions)
  total_q = len(questions)

  print(f"\n{'='*60}")
  print(f"  Cost estimate — {model_id}  ({total_q} questions)")
  print(f"{'='*60}")

  # ---- CSV token summary ----
  print("\nCSV sizes:")
  total_csv_no_cache = 0
  total_csv_with_cache = 0
  csv_write_tokens = 0

  for db_id, n in sorted(count_by_db.items()):
    csv_tok = _csv_tokens_for_db(db_id)
    no_cache = csv_tok * n
    total_csv_no_cache += no_cache

    # With cache: 1 write + (n-1) reads
    write = csv_tok
    reads = csv_tok * (n - 1) if n > 1 else 0
    total_csv_with_cache += write + reads
    csv_write_tokens += write

    long_flag = " ⚠ >200K (Gemini tier-2 pricing)" if csv_tok > LONG_CONTEXT_THRESHOLD else ""
    print(f"  {db_id:<35}: {csv_tok:>8,} tok × {n:>4} questions{long_flag}")

  total_prompt = avg_prompt_tokens * total_q
  total_output = avg_output_tokens * total_q

  # ---- Cost helpers ----
  def _input_price(tok: int, db_id: str | None = None) -> float:
    """Price for `tok` standard (non-cached) input tokens in USD."""
    per_1m = pricing.input
    if db_id and pricing.input_long is not None:
      csv_tok = _csv_tokens_for_db(db_id)
      if csv_tok > LONG_CONTEXT_THRESHOLD:
        per_1m = pricing.input_long
    return tok * per_1m / 1_000_000

  def _cache_read_price(tok: int, long: bool = False) -> float:
    per_1m = pricing.cache_read_long if (long and pricing.cache_read_long) else pricing.cache_read
    return tok * per_1m / 1_000_000

  def _cache_write_price(tok: int) -> float:
    return tok * pricing.input * pricing.cache_write_mult / 1_000_000

  def _output_price(tok: int) -> float:
    return tok * pricing.output / 1_000_000

  # ---- Scenario calculations ----
  def _scenario(batch: bool, cache: bool) -> float:
    bd = pricing.batch_discount if batch else 1.0

    if cache:
      csv_cost = (
        _cache_write_price(csv_write_tokens) * bd
        + _cache_read_price(total_csv_with_cache - csv_write_tokens) * bd
      )
    else:
      csv_cost = _input_price(total_csv_no_cache) * bd

    prompt_cost = _input_price(total_prompt) * bd
    output_cost = _output_price(total_output) * bd
    return csv_cost + prompt_cost + output_cost

  no_opt    = _scenario(batch=False, cache=False)
  batch_only = _scenario(batch=True,  cache=False)
  cache_only = _scenario(batch=False, cache=True)
  both       = _scenario(batch=True,  cache=True)

  print(f"\nCost scenarios:")
  print(f"  No optimization   : ${no_opt:.2f}")
  print(f"  Batch only  (-50%): ${batch_only:.2f}")
  print(f"  Cache only  (-90%): ${cache_only:.2f}")

  rec = " ← recommended" if use_batch and use_cache else ""
  print(f"  Batch + Cache     : ${both:.2f}{rec}")

  if use_batch and use_cache:
    active = both
  elif use_batch:
    active = batch_only
  elif use_cache:
    active = cache_only
  else:
    active = no_opt

  flags = " + ".join(filter(None, ["batch" if use_batch else "", "cache" if use_cache else ""])) or "none"
  print(f"\n  Active config ({flags}): ${active:.2f}")
  print(f"{'='*60}\n")


def estimate_all(
  questions: list["Question"],
  model_configs: list[tuple[str, str]],  # list of (provider, model_id)
  **kwargs,
) -> None:
  """Estimate for multiple models at once."""
  total = 0.0
  for _provider, model_id in model_configs:
    estimate(questions, model_id, **kwargs)
