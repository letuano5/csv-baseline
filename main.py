"""
CSV Baseline — entry point.

Usage:
  # Estimate cost only (no API calls)
  uv run main.py --estimate
  uv run main.py --provider claude --model-id claude-sonnet-4-6 --checkpoint run-01 --estimate

  # Full run
  uv run main.py --provider claude  --model-id claude-sonnet-4-6           --checkpoint run-01
  uv run main.py --provider gemini  --model-id gemini-2.5-pro-preview       --checkpoint run-01
  uv run main.py --provider openai  --model-id gpt-5.4                      --checkpoint run-01

  # Smoke-test with first N questions
  uv run main.py --provider claude --model-id claude-sonnet-4-6 --checkpoint run-01 --limit 5

  # Export selected questions to file (no API calls)
  uv run main.py --limit 1000 --export-questions selected_questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import QUESTIONS_PATH, get_csv_meta
from estimator import estimate, PRICING
from runners.base import load_questions
from runners.claude_runner import ClaudeRunner
from runners.gemini_runner import GeminiRunner
from runners.openai_runner import OpenAIRunner

_RUNNER_REGISTRY = {
  "claude": ClaudeRunner,
  "gemini": GeminiRunner,
  "openai": OpenAIRunner,
}

# Default model for each provider (used by --estimate without --model-id)
_DEFAULT_MODELS: dict[str, str] = {
  "claude": "claude-sonnet-4-6",
  "gemini": "gemini-3.1-pro-preview",
  "openai": "gpt-5.4",
}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="CSV baseline evaluation")
  p.add_argument("--provider", choices=list(_RUNNER_REGISTRY), help="API provider")
  p.add_argument("--model-id", help="Model identifier passed to the API")
  p.add_argument("--checkpoint", help="Output sub-directory name, e.g. run-01")
  p.add_argument("--questions", default=str(QUESTIONS_PATH), help="Path to questions.json")
  p.add_argument("--limit", type=int, default=None, help="Only process first N questions")
  p.add_argument(
    "--estimate",
    action="store_true",
    help="Print cost estimate and exit (no API calls)",
  )
  p.add_argument(
    "--export-questions",
    metavar="OUTPUT_PATH",
    help="Export selected questions (after sort + limit) to a JSON file and exit",
  )
  p.add_argument(
    "--retry-errors",
    action="store_true",
    help="Re-run questions whose saved result starts with ERROR: (e.g. after a parser fix)",
  )
  p.add_argument(
    "--resume-batch",
    metavar="BATCH_ID",
    help="Retrieve results from an already-submitted batch (Claude: msgbatch_01..., OpenAI: batch_...)",
  )
  p.add_argument(
    "--resume-from-file",
    metavar="JSONL_PATH",
    help="Parse a locally-downloaded batch output JSONL file and merge into checkpoint (OpenAI only)",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  questions_path = Path(args.questions)

  # Load questions — sort by schema CSV size (smallest first) then apply limit
  questions = load_questions(questions_path)
  questions.sort(key=lambda q: get_csv_meta(q.db_id).size_bytes)
  if args.limit:
    questions = questions[: args.limit]

  # ---- Export questions mode ----
  if args.export_questions:
    out_path = Path(args.export_questions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
      json.dumps([asdict(q) for q in questions], ensure_ascii=False, indent=2),
      encoding="utf-8",
    )
    print(f"Exported {len(questions)} questions → {out_path}")
    return

  # ---- Estimate mode ----
  if args.estimate:
    if args.provider and args.model_id:
      estimate(questions, args.model_id)
    elif args.provider:
      model_id = _DEFAULT_MODELS.get(args.provider, args.provider)
      estimate(questions, model_id)
    else:
      # Estimate all default models
      for _provider, model_id in _DEFAULT_MODELS.items():
        estimate(questions, model_id)
    return

  # ---- Run mode ----
  if not args.provider:
    print("Error: --provider is required for run mode.", file=sys.stderr)
    sys.exit(1)
  if not args.checkpoint:
    print("Error: --checkpoint is required for run mode.", file=sys.stderr)
    sys.exit(1)

  model_id = args.model_id or _DEFAULT_MODELS.get(args.provider)
  if not model_id:
    print(f"Error: --model-id is required for provider {args.provider!r}.", file=sys.stderr)
    sys.exit(1)

  runner_cls = _RUNNER_REGISTRY[args.provider]
  runner = runner_cls(model_id=model_id, checkpoint_name=args.checkpoint)

  print(f"Provider : {args.provider}")
  print(f"Model    : {model_id}")
  print(f"Questions: {len(questions)} (from {questions_path})")
  print(f"Output   : output/{args.checkpoint}/{model_id}.json")
  print()

  if args.resume_from_file:
    if not hasattr(runner, "resume_from_file"):
      print(f"Error: --resume-from-file not supported for --provider {args.provider}", file=sys.stderr)
      sys.exit(1)
    runner.resume_from_file(args.resume_from_file, questions, retry_errors=args.retry_errors)
    return

  if args.resume_batch:
    if not hasattr(runner, "resume_batch"):
      print(f"Error: --resume-batch not supported for --provider {args.provider}", file=sys.stderr)
      sys.exit(1)
    runner.resume_batch(args.resume_batch, questions, retry_errors=args.retry_errors)
    return

  results = runner.run(questions, retry_errors=args.retry_errors)

  errors = sum(1 for r in results if r.result.startswith("ERROR:"))
  print(f"\nDone. {len(results)} answers, {errors} errors.")
  print(f"Output: output/{args.checkpoint}/{model_id}.json")


if __name__ == "__main__":
  main()
