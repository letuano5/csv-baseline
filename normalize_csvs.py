"""
Normalize CSV files under input/csv and optionally write in-place.

Rules:
- Encoding: output utf-8 (no BOM)
- Delimiter: output comma
- Header: trim whitespace, remove BOM, strip Vietnamese diacritics to ASCII snake_case
- Missing values: normalize common markers to empty string
- Dates: normalize mixed date strings to YYYY-MM-DD when parseable
- Numeric values: normalize numeric-like strings in numeric-like columns
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CSV_DIR = Path("input/csv")
REPORT_PATH = Path("output/normalization_report.json")

MISSING_MARKERS = {"na", "n/a", "null", "none", "nan", "-", "--"}

_DATE_FORMATS = (
  "%Y-%m-%d",
  "%Y/%m/%d",
  "%d/%m/%Y",
  "%d/%m/%y",
  "%d-%m-%Y",
  "%d-%m-%y",
  "%Y-%m-%d %H:%M:%S",
  "%Y-%m-%d %H:%M",
  "%Y-%m-%dT%H:%M:%S",
  "%Y-%m-%dT%H:%M",
)

_NUMERIC_HINTS = (
  "year",
  "ratio",
  "amount",
  "value",
  "score",
  "price",
  "cost",
  "revenue",
  "income",
  "asset",
  "deposit",
  "loan",
  "profit",
  "loss",
  "percent",
  "percentage",
  "latitude",
  "longitude",
  "capacity",
  "stt",
)


@dataclass
class FileStats:
  file: str
  rows: int = 0
  cols: int = 0
  encoding_in: str = "utf-8"
  delimiter_in: str = ","
  headers_trimmed: int = 0
  missing_normalized: int = 0
  dates_normalized: int = 0
  numerics_normalized: int = 0
  output_path: str = ""


def detect_encoding(raw: bytes) -> str:
  return "utf-8-sig" if raw[:3] == b"\xef\xbb\xbf" else "utf-8"


def sniff_delimiter(sample: str) -> str:
  try:
    dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|")
    return dialect.delimiter
  except csv.Error:
    return ","


def normalize_header(name: str) -> str:
  return (name or "").replace("\ufeff", "").strip()


def is_missing(v: str) -> bool:
  s = (v or "").strip()
  return s == "" or s.lower() in MISSING_MARKERS


def normalize_date(value: str) -> str | None:
  s = value.strip()
  if not s:
    return None
  s = s.rstrip(".")

  for fmt in _DATE_FORMATS:
    try:
      dt = datetime.strptime(s, fmt)
      return dt.strftime("%Y-%m-%d")
    except ValueError:
      continue
  return None


def _looks_numeric(s: str) -> bool:
  return bool(re.match(r"^[-+]?\d+([.,]\d+)?$", s))


def _normalize_numeric_string(raw: str) -> str | None:
  s = raw.strip()
  if not s:
    return None

  had_percent = "%" in s
  s = re.sub(r"[₫$€£]", "", s)
  s = re.sub(r"\b(VND|USD|EUR|GBP)\b", "", s, flags=re.IGNORECASE)
  s = s.replace("%", "").strip()

  if re.match(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$", s):
    s = s.replace(",", "")
  elif re.match(r"^[-+]?\d+,\d+$", s):
    s = s.replace(",", ".")

  if not _looks_numeric(s):
    return None

  num = float(s)
  if num.is_integer():
    out = str(int(num))
  else:
    out = f"{num:.10f}".rstrip("0").rstrip(".")

  # Keep explicit percentage semantics as numeric value only.
  return out if not had_percent else out


def pick_numeric_columns(
  headers: list[str],
  row_keys: list[str],
  rows: list[dict[str, str]],
) -> set[str]:
  """Detect numeric-like columns. `row_keys` are DictReader keys; `headers` are output (slug) names."""
  numeric_cols: set[str] = set()
  sample_size = min(1000, len(rows))
  sample_rows = rows[:sample_size]

  for h, rk in zip(headers, row_keys):
    h_low = h.lower()
    hint = any(token in h_low for token in _NUMERIC_HINTS)
    non_missing = 0
    numeric_like = 0
    for row in sample_rows:
      v = (row.get(rk) or "").strip()
      if not v or is_missing(v):
        continue
      non_missing += 1
      if _normalize_numeric_string(v) is not None:
        numeric_like += 1
    ratio = (numeric_like / non_missing) if non_missing else 0.0
    if hint and ratio >= 0.6:
      numeric_cols.add(h)
    elif ratio >= 0.95 and non_missing >= 20:
      numeric_cols.add(h)
  return numeric_cols


def _slugify_header(name: str) -> str:
  s = (name or "").strip().lower()
  s = unicodedata.normalize("NFKD", s)
  s = "".join(ch for ch in s if not unicodedata.combining(ch))
  s = s.replace("đ", "d")
  s = re.sub(r"[^a-z0-9]+", "_", s)
  return s.strip("_")


def _unique_slug_headers(names: list[str]) -> list[str]:
  """One slug per column position; append _2, _3, ... on collisions."""
  out: list[str] = []
  seen: dict[str, int] = {}
  for n in names:
    base = _slugify_header(n)
    if base not in seen:
      seen[base] = 1
      out.append(base)
      continue
    seen[base] += 1
    out.append(f"{base}_{seen[base]}")
  return out


_FACILITY_DATE_NOTE_RE = re.compile(r"\((.*?)\)\s*$")
_FACILITY_SYT_DATE_RE = re.compile(
  r"syt(?:\s+\w+){0,3}?\s*(?:nhan|tiep\s*nhan)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
  re.IGNORECASE,
)


def _extract_facility_date_parts(raw_value: str) -> tuple[str, str, str]:
  """
  Return tuple: (ngay_cong_bo_iso, ngay_syt_nhan_iso, ghi_chu_ngay_cong_bo).
  Handles values like:
  - "13/1/2025."
  - "6/12/2024 (SYT nhận 9/1/2025)"
  - "... (some other note)"
  """
  s = raw_value.strip()
  if not s:
    return "", "", ""

  note = ""
  m_note = _FACILITY_DATE_NOTE_RE.search(s)
  if m_note:
    note = m_note.group(1).strip()
    s = s[: m_note.start()].strip()

  primary_date = normalize_date(s) or ""
  syt_date = ""
  if note:
    normalized_note = (
      unicodedata.normalize("NFKD", note)
      .encode("ascii", "ignore")
      .decode("ascii")
      .lower()
    )
    m_syt = _FACILITY_SYT_DATE_RE.search(normalized_note)
    if m_syt:
      syt_date = normalize_date(m_syt.group(1)) or ""

  # Keep note only when it is not solely a clean "SYT nhận <date>" pattern.
  note_clean = note
  if note:
    stripped_note = re.sub(r"\s+", " ", note).strip().lower()
    stripped_note = (
      unicodedata.normalize("NFKD", stripped_note)
      .encode("ascii", "ignore")
      .decode("ascii")
    )
    if _FACILITY_SYT_DATE_RE.search(stripped_note):
      stripped_note = _FACILITY_SYT_DATE_RE.sub("", stripped_note).strip(" -:,;")
      if not stripped_note:
        note_clean = ""

  return primary_date, syt_date, note_clean


def _postprocess_facility(headers: list[str], rows: list[dict[str, str]], stats: FileStats) -> tuple[list[str], list[dict[str, str]]]:
  if "ngay_cong_bo" not in headers:
    return headers, rows

  new_headers = list(headers)
  idx = new_headers.index("ngay_cong_bo")
  if "ngay_syt_nhan" not in new_headers:
    new_headers.insert(idx + 1, "ngay_syt_nhan")
  if "ghi_chu_ngay_cong_bo" not in new_headers:
    idx = new_headers.index("ngay_cong_bo")
    insert_pos = idx + 2 if "ngay_syt_nhan" in new_headers else idx + 1
    new_headers.insert(insert_pos, "ghi_chu_ngay_cong_bo")

  out_rows: list[dict[str, str]] = []
  for row in rows:
    ngay_cb, ngay_syt, note = _extract_facility_date_parts(row.get("ngay_cong_bo", ""))
    existing_note = (row.get("ghi_chu_ngay_cong_bo", "") or "").strip()
    existing_syt = (row.get("ngay_syt_nhan", "") or "").strip()
    if not ngay_syt and existing_syt:
      ngay_syt = normalize_date(existing_syt) or existing_syt
    if existing_note:
      # Second-pass extraction for files already normalized previously.
      _, note_syt, _ = _extract_facility_date_parts(f"2000-01-01 ({existing_note})")
      if not ngay_syt and note_syt:
        ngay_syt = note_syt

    current = dict(row)
    if current.get("ngay_cong_bo", "") != ngay_cb:
      stats.dates_normalized += 1
    current["ngay_cong_bo"] = ngay_cb
    current["ngay_syt_nhan"] = ngay_syt
    current["ghi_chu_ngay_cong_bo"] = note
    out_rows.append(current)
  return new_headers, out_rows


def normalize_file(path: Path, inplace: bool) -> FileStats:
  raw = path.read_bytes()
  encoding = detect_encoding(raw)
  text = raw.decode(encoding, errors="replace")
  delimiter = sniff_delimiter(text)

  reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
  headers_in = reader.fieldnames or []
  headers_trimmed = [normalize_header(h) for h in headers_in]
  headers = _unique_slug_headers(headers_trimmed)
  rows_in = list(reader)

  stats = FileStats(
    file=path.name,
    rows=len(rows_in),
    cols=len(headers),
    encoding_in=encoding,
    delimiter_in=delimiter,
    headers_trimmed=sum(1 for a, b in zip(headers_in, headers) if (a or "") != b),
  )

  numeric_cols = pick_numeric_columns(headers, headers_in, rows_in)

  cleaned_rows: list[dict[str, str]] = []
  for row in rows_in:
    cleaned: dict[str, str] = {}
    for old_h, new_h in zip(headers_in, headers):
      raw_val = row.get(old_h, "")
      val = "" if raw_val is None else str(raw_val)

      if is_missing(val):
        if val.strip() != "":
          stats.missing_normalized += 1
        cleaned[new_h] = ""
        continue

      stripped = val.strip()

      date_val = normalize_date(stripped)
      if date_val is not None and date_val != stripped:
        cleaned[new_h] = date_val
        stats.dates_normalized += 1
        continue

      if new_h in numeric_cols:
        numeric_val = _normalize_numeric_string(stripped)
        if numeric_val is not None:
          if numeric_val != stripped:
            stats.numerics_normalized += 1
          cleaned[new_h] = numeric_val
          continue

      cleaned[new_h] = stripped
    cleaned_rows.append(cleaned)

  if path.name == "facility_announcements_2025.csv":
    headers, cleaned_rows = _postprocess_facility(headers, cleaned_rows, stats)

  if inplace:
    out_path = path
  else:
    out_path = path.with_stem(path.stem + ".normalized")
  stats.output_path = str(out_path)

  with out_path.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=headers, delimiter=",")
    writer.writeheader()
    writer.writerows(cleaned_rows)

  return stats


def main() -> None:
  parser = argparse.ArgumentParser(description="Normalize CSV files in input/csv")
  parser.add_argument(
    "--inplace",
    action="store_true",
    help="Overwrite original CSV files in-place",
  )
  parser.add_argument(
    "--glob",
    default="*.csv",
    help="Glob pattern inside input/csv (default: *.csv)",
  )
  args = parser.parse_args()

  files = sorted([p for p in CSV_DIR.glob(args.glob) if p.is_file() and not p.name.startswith(".")])
  if not files:
    raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")

  report: list[dict] = []
  for path in files:
    stats = normalize_file(path, inplace=args.inplace)
    report.append(stats.__dict__)
    print(
      f"{path.name}: rows={stats.rows}, headers_trimmed={stats.headers_trimmed}, "
      f"missing={stats.missing_normalized}, dates={stats.dates_normalized}, "
      f"numerics={stats.numerics_normalized}, out={stats.output_path}"
    )

  REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
  REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
  main()
