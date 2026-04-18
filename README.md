# Baseline Test

Run all:
```
uv run main.py --provider claude --checkpoint run-light --limit 300 --retry-errors
```

Get result from batch:
```
uv run main.py --provider claude --checkpoint run-light --resume-batch msgbatch_01xxxxx
```