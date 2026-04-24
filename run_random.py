from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from pathlib import Path


def _sanitize_name(value: str) -> str:
  text = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
  return text.strip("-").lower() or "model"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Randomly select questions, clean old output, and run main.py."
  )
  p.add_argument("--model-id", required=True, help="Model id, e.g. qwen/qwen3.6-plus")
  p.add_argument("--limit", required=True, type=int, help="Number of random questions")
  p.add_argument("--provider", default="openrouter", help="Provider for main.py")
  p.add_argument(
    "--questions-source",
    default="input/questions.json",
    help="Source question pool JSON file",
  )
  p.add_argument(
    "--selected-file",
    default=None,
    help="Path to save selected random questions (default: selected_random_<limit>.json)",
  )
  p.add_argument(
    "--checkpoint",
    default=None,
    help="Checkpoint name (default: run-<provider>-<model>-random-<limit>)",
  )
  p.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional random seed for reproducible sampling",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  if args.limit <= 0:
    raise SystemExit("--limit must be > 0")

  source_path = Path(args.questions_source)
  if not source_path.exists():
    raise SystemExit(f"Questions source not found: {source_path}")

  questions = json.loads(source_path.read_text(encoding="utf-8"))
  if not isinstance(questions, list):
    raise SystemExit(f"Invalid questions format (expected list): {source_path}")
  if args.limit > len(questions):
    raise SystemExit(f"--limit={args.limit} is larger than source size={len(questions)}")

  if args.seed is not None:
    random.seed(args.seed)

  selected = random.sample(questions, args.limit)

  model_slug = _sanitize_name(args.model_id.replace("/", "-"))
  checkpoint = args.checkpoint or f"run-{args.provider}-{model_slug}-random-{args.limit}"
  selected_file = Path(args.selected_file or f"selected_random_{args.limit}.json")

  # 1) Clean old checkpoint output
  output_dir = Path("output") / checkpoint
  if output_dir.exists():
    shutil.rmtree(output_dir)
    print(f"Deleted old output: {output_dir}")
  else:
    print(f"No old output to delete: {output_dir}")

  # 2) Save selected random questions
  selected_file.write_text(
    json.dumps(selected, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"Saved random selection: {selected_file} ({args.limit} questions)")

  # 3) Run main.py with selected questions
  cmd = [
    "uv",
    "run",
    "main.py",
    "--provider",
    args.provider,
    "--model-id",
    args.model_id,
    "--questions",
    str(selected_file),
    "--checkpoint",
    checkpoint,
  ]
  print("Running:", " ".join(cmd))
  subprocess.run(cmd, check=True)


if __name__ == "__main__":
  main()
