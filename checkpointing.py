"""
Checkpoint: load / save answered questions to output/<checkpoint>/<model_id>.json.

The file is an ordered JSON array. Each element is an AnsweredQuestion dict.
The file is fully rewritten on every save (atomic-ish — small files, fast I/O).
"""

import json
from dataclasses import asdict
from pathlib import Path

from config import OUTPUT_DIR


def checkpoint_path(checkpoint_name: str, model_id: str) -> Path:
  return OUTPUT_DIR / checkpoint_name / f"{model_id}.json"


def load_checkpoint(checkpoint_name: str, model_id: str) -> list[dict]:
  """Return list of already-answered question dicts, or [] if file not found."""
  path = checkpoint_path(checkpoint_name, model_id)
  if not path.exists():
    return []
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError):
    return []


def save_checkpoint(
  checkpoint_name: str,
  model_id: str,
  results: list,  # list of AnsweredQuestion or dict
) -> None:
  path = checkpoint_path(checkpoint_name, model_id)
  path.parent.mkdir(parents=True, exist_ok=True)
  serializable = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results]
  path.write_text(
    json.dumps(serializable, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )


def answered_indices(
  checkpoint_name: str,
  model_id: str,
  retry_errors: bool = False,
) -> set[int]:
  """
  Return the set of question indices that are already done.

  Args:
    retry_errors: If True, indices whose saved result starts with "ERROR:"
                  are excluded from the done set so they get re-run.
  """
  records = load_checkpoint(checkpoint_name, model_id)
  if retry_errors:
    return {r["index"] for r in records if not r.get("result", "").startswith("ERROR:")}
  return {r["index"] for r in records}
