"""
config.py — paths, constants, and CSV metadata.

CsvMeta is built lazily from the actual files on disk: encoding and delimiter
are auto-detected; column names are read from the header row. Nothing is
hardcoded — if you swap out a CSV file the metadata updates automatically.

db_id  →  filename convention:  {db_id}.csv  (same as the questions.json field)
"""

import csv
import io
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
QUESTIONS_PATH = ROOT / "input" / "questions.json"
CSV_DIR = ROOT / "input" / "csv"
OUTPUT_DIR = ROOT / "output"

# Batch / concurrency settings
MINI_BATCH_SIZE = 10_000
GEMINI_CONCURRENCY = 20
OPENROUTER_CONCURRENCY = 20
POLL_INTERVAL_SECONDS = 30
POLL_TIMEOUT_SECONDS = 86_400  # 24 h

# Estimated output tokens per question (model-generated code + explanation)
AVG_OUTPUT_TOKENS = 800

# Estimated non-CSV input tokens per question (system prompt + schema + question)
AVG_PROMPT_TOKENS = 600


@dataclass
class CsvMeta:
  db_id: str
  filename: str
  delimiter: str
  encoding: str
  columns: list[str]

  @property
  def path(self) -> Path:
    return CSV_DIR / self.filename

  @property
  def size_bytes(self) -> int:
    return self.path.stat().st_size

  @property
  def approx_tokens(self) -> int:
    # 4 bytes per token is a stable approximation for CSV content across all providers.
    return self.size_bytes // 4


def _detect_encoding(raw: bytes) -> str:
  """Return 'utf-8-sig' if BOM present, else 'utf-8'."""
  return "utf-8-sig" if raw[:3] == b"\xef\xbb\xbf" else "utf-8"


def _detect_delimiter(sample: str) -> str:
  """Use csv.Sniffer on the first ~8 KB of text; fall back to comma."""
  try:
    dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|")
    return dialect.delimiter
  except csv.Error:
    return ","


def _read_header(path: Path, encoding: str, delimiter: str) -> list[str]:
  """Read the first row of the CSV and return column names (stripped)."""
  with open(path, encoding=encoding, newline="") as fh:
    reader = csv.reader(fh, delimiter=delimiter)
    return [col.strip() for col in next(reader)]


@lru_cache(maxsize=None)
def get_csv_meta(db_id: str) -> CsvMeta:
  """
  Build CsvMeta for `db_id` by inspecting the file on disk.
  Result is cached so the file is only read once per process.
  """
  filename = f"{db_id}.csv"
  path = CSV_DIR / filename
  if not path.exists():
    raise FileNotFoundError(f"CSV file not found: {path}")

  raw = path.read_bytes()
  encoding = _detect_encoding(raw)
  sample = raw.decode(encoding, errors="replace")
  delimiter = _detect_delimiter(sample)
  columns = _read_header(path, encoding, delimiter)

  return CsvMeta(
    db_id=db_id,
    filename=filename,
    delimiter=delimiter,
    encoding=encoding,
    columns=columns,
  )


# ---------------------------------------------------------------------------
# Convenience proxy — keeps old CSV_REGISTRY[db_id] call sites working
# ---------------------------------------------------------------------------

class _LazyRegistry:
  """Dict-like proxy that calls get_csv_meta() on first access per key."""

  def __getitem__(self, db_id: str) -> CsvMeta:
    return get_csv_meta(db_id)

  def __contains__(self, db_id: str) -> bool:
    return (CSV_DIR / f"{db_id}.csv").exists()


CSV_REGISTRY = _LazyRegistry()
