"""
Generate dataset profiles from local CSV files (no external API).

Usage:
  uv run generate_profiles.py
  uv run generate_profiles.py --db-id facility_announcements_2025
  uv run generate_profiles.py --input-dir input/csv --output-dir profiles/auto
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MISSING_MARKERS = {"", "na", "n/a", "null", "none", "nan", "-", "--"}
BOOL_MARKERS = {"0", "1", "true", "false", "yes", "no", "y", "n"}
DATE_PATTERNS = (
  re.compile(r"^\d{4}-\d{2}-\d{2}$"),
  re.compile(r"^\d{4}/\d{2}/\d{2}$"),
  re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
  re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),
  re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(:\d{2})?$"),
)
NUMERIC_RE = re.compile(r"^[-+]?\d+(\.\d+)?$")
THOUSAND_NUMERIC_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$")
COMMA_DECIMAL_RE = re.compile(r"^[-+]?\d+,\d+$")
ID_HINTS = ("id", "code", "ma", "sbd", "stt")


@dataclass
class ColumnStats:
  missing_count: int
  missing_ratio: float
  unique_count: int
  unique_ratio: float
  top_values: list[tuple[str, int]]
  sample_values: list[str]
  date_match_ratio: float
  numeric_match_ratio: float
  bool_match_ratio: float


def detect_encoding(raw: bytes) -> str:
  return "utf-8-sig" if raw[:3] == b"\xef\xbb\xbf" else "utf-8"


def sniff_delimiter(sample: str) -> str:
  try:
    dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|")
    return dialect.delimiter
  except csv.Error:
    return ","


def normalize_value(v: str | None) -> str:
  if v is None:
    return ""
  return str(v).strip()


def is_missing(v: str) -> bool:
  return v.lower() in MISSING_MARKERS


def looks_date(v: str) -> bool:
  return any(p.match(v) for p in DATE_PATTERNS)


def normalize_numeric_for_check(v: str) -> str | None:
  s = v.strip()
  if not s:
    return None
  s = re.sub(r"[₫$€£%]", "", s)
  s = re.sub(r"\b(VND|USD|EUR|GBP)\b", "", s, flags=re.IGNORECASE).strip()
  if THOUSAND_NUMERIC_RE.match(s):
    s = s.replace(",", "")
  elif COMMA_DECIMAL_RE.match(s):
    s = s.replace(",", ".")
  return s if NUMERIC_RE.match(s) else None


def infer_column_stats(values: list[str], total_rows: int) -> ColumnStats:
  normalized = [normalize_value(v) for v in values]
  non_missing = [v for v in normalized if not is_missing(v)]
  missing_count = total_rows - len(non_missing)
  missing_ratio = (missing_count / total_rows) if total_rows else 0.0

  unique_count = len(set(non_missing))
  unique_ratio = (unique_count / len(non_missing)) if non_missing else 0.0

  counter = Counter(non_missing)
  top_values = counter.most_common(5)
  sample_values = list(dict.fromkeys(non_missing))[:5]

  date_hits = sum(1 for v in non_missing if looks_date(v))
  numeric_hits = sum(1 for v in non_missing if normalize_numeric_for_check(v) is not None)
  bool_hits = sum(1 for v in non_missing if v.lower() in BOOL_MARKERS)
  denom = max(1, len(non_missing))

  return ColumnStats(
    missing_count=missing_count,
    missing_ratio=round(missing_ratio, 6),
    unique_count=unique_count,
    unique_ratio=round(unique_ratio, 6),
    top_values=top_values,
    sample_values=sample_values,
    date_match_ratio=round(date_hits / denom, 6),
    numeric_match_ratio=round(numeric_hits / denom, 6),
    bool_match_ratio=round(bool_hits / denom, 6),
  )


# Classification thresholds below were tuned on the Vietnamese CSV corpus in input/csv/.
# If a new dataset misfires, adjust the ratio thresholds rather than adding special cases.
def infer_semantic_groups(column_stats: dict[str, ColumnStats]) -> dict[str, list[str]]:
  date_columns: list[str] = []
  numeric_columns: list[str] = []
  boolean_like_columns: list[str] = []
  id_like_columns: list[str] = []
  high_missing_columns: list[str] = []
  numeric_as_text_columns: list[str] = []

  for col, stats in column_stats.items():
    low = col.lower()
    if stats.date_match_ratio >= 0.7:
      date_columns.append(col)
    if stats.numeric_match_ratio >= 0.85 and stats.unique_count >= 5:
      numeric_columns.append(col)
    if stats.bool_match_ratio >= 0.95:
      boolean_like_columns.append(col)
    if stats.missing_ratio > 0.3:
      high_missing_columns.append(col)
    if stats.numeric_match_ratio >= 0.6 and stats.numeric_match_ratio < 0.85:
      numeric_as_text_columns.append(col)
    if stats.unique_ratio > 0.95 and any(h in low for h in ID_HINTS):
      id_like_columns.append(col)

  return {
    "date_columns": sorted(date_columns),
    "numeric_columns": sorted(numeric_columns),
    "boolean_like_columns": sorted(boolean_like_columns),
    "id_like_columns": sorted(id_like_columns),
    "high_missing_columns": sorted(high_missing_columns),
    "numeric_as_text_columns": sorted(numeric_as_text_columns),
  }


def profile_csv(path: Path) -> dict:
  raw = path.read_bytes()
  encoding = detect_encoding(raw)
  text = raw.decode(encoding, errors="replace")
  delimiter = sniff_delimiter(text)

  reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
  headers = reader.fieldnames or []
  rows = list(reader)
  row_count = len(rows)

  per_col_values: dict[str, list[str]] = {h: [] for h in headers}
  for row in rows:
    for h in headers:
      per_col_values[h].append(normalize_value(row.get(h)))

  col_stats = {
    h: infer_column_stats(per_col_values[h], row_count)
    for h in headers
  }
  groups = infer_semantic_groups(col_stats)

  mixed_date_format_columns: list[str] = []
  for h in groups["date_columns"]:
    vals = [v for v in per_col_values[h] if not is_missing(v)]
    formats = set()
    for v in vals[:5000]:
      if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        formats.add("yyyy-mm-dd")
      elif re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", v):
        formats.add("d/m/y")
      elif re.match(r"^\d{1,2}-\d{1,2}-\d{2,4}$", v):
        formats.add("d-m-y")
      elif re.match(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(:\d{2})?$", v):
        formats.add("iso-datetime")
    if len(formats) > 1:
      mixed_date_format_columns.append(h)

  profile = {
    "db_id": path.stem,
    "file_path": str(path),
    "encoding": encoding,
    "delimiter": delimiter,
    "row_count": row_count,
    "column_count": len(headers),
    "columns": headers,
    "inferred_types": groups,
    "quality_flags": {
      "mixed_date_format_columns": sorted(mixed_date_format_columns),
      "has_bom": encoding == "utf-8-sig",
      "non_comma_delimiter": delimiter != ",",
    },
    "column_stats": {
      h: {
        "missing_count": s.missing_count,
        "missing_ratio": s.missing_ratio,
        "unique_count": s.unique_count,
        "unique_ratio": s.unique_ratio,
        "top_values": s.top_values,
        "sample_values": s.sample_values,
        "date_match_ratio": s.date_match_ratio,
        "numeric_match_ratio": s.numeric_match_ratio,
        "bool_match_ratio": s.bool_match_ratio,
      }
      for h, s in col_stats.items()
    },
  }
  return profile


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Generate local dataset profiles from CSV files.")
  p.add_argument("--input-dir", default="input/csv", help="Directory containing CSV files.")
  p.add_argument("--output-dir", default="profiles/auto", help="Directory to write profile JSON files.")
  p.add_argument("--db-id", help="Generate profile for one db_id only (CSV filename without .csv).")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  if args.db_id:
    files = [input_dir / f"{args.db_id}.csv"]
  else:
    files = sorted(input_dir.glob("*.csv"))

  if not files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

  generated = 0
  for csv_file in files:
    if not csv_file.exists():
      print(f"Skip missing file: {csv_file}")
      continue
    profile = profile_csv(csv_file)
    out = output_dir / f"{csv_file.stem}.json"
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    generated += 1
    print(f"Generated profile: {out}")

  print(f"\nDone. Generated {generated} profile(s) in {output_dir}")


if __name__ == "__main__":
  main()
