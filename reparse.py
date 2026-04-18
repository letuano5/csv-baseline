"""
Re-parse an existing output JSON using the current result_parser logic,
without making any API calls.

Usage:
  uv run reparse.py output/run-light/claude-sonnet-4-6.json
  uv run reparse.py output/run-light/claude-sonnet-4-6.json --inplace
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from result_parser import _extract_json_from_text

_OUTPUT_RE = re.compile(r"\[OUTPUT\](.*?)\[/OUTPUT\]", re.DOTALL)


def _stdout_from_raw(raw_output: str) -> str:
  """Extract the [OUTPUT] section if present, otherwise use the full string."""
  m = _OUTPUT_RE.search(raw_output)
  return m.group(1) if m else raw_output


def reparse(path: Path, inplace: bool) -> None:
  with open(path, encoding="utf-8") as f:
    data: list[dict] = json.load(f)

  changed = 0
  for entry in data:
    raw = entry.get("raw_output", "")
    if not raw:
      continue
    old_result = entry.get("result", "")
    stdout = _stdout_from_raw(raw)
    new_result = _extract_json_from_text(stdout)
    if new_result != old_result:
      entry["result"] = new_result
      changed += 1

  print(f"Re-parsed {len(data)} entries — {changed} changed.")

  if inplace:
    out_path = path
  else:
    out_path = path.with_stem(path.stem + ".reparsed")

  out_path.write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"Written → {out_path}")

  # Summary of new ERROR counts
  errors = sum(1 for e in data if e.get("result", "").startswith("ERROR:"))
  tool_limit = sum(1 for e in data if e.get("result", "") == "ERROR:tool_limit_exceeded")
  print(f"Total ERROR entries: {errors}  (of which tool_limit_exceeded: {tool_limit})")


def main() -> None:
  p = argparse.ArgumentParser(description="Re-parse output JSON with updated result_parser")
  p.add_argument("path", help="Path to the output JSON file")
  p.add_argument("--inplace", action="store_true", help="Overwrite the original file")
  args = p.parse_args()
  reparse(Path(args.path), inplace=args.inplace)


if __name__ == "__main__":
  main()
