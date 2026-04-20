# Baseline Test

Run all:
```bash
uv run main.py --provider claude --checkpoint run-light --limit 300 --retry-errors
```

Get result from batch:
```bash
uv run main.py --provider claude --checkpoint run-light --resume-batch msgbatch_01xxxxx
```

## Baseline Pipeline

```text
[0] OPTIONAL DATA PREP
+-----------------+      +--------------------+      +----------------------+
| input/csv/*.csv | ---> | normalize_csvs.py  | ---> | generate_profiles.py |
+-----------------+      +--------------------+      +----------------------+
                                                               |
                                                     +----------------------+
                                                     | profiles/auto/*.json |
                                                     +----------------------+

[1] ENTRY + MAIN FLOW
+----------------+      +----------------+      +------------------+      +------------------+
| uv run main.py | ---> | parse CLI args | ---> | load questions   | ---> | apply --limit    |
+----------------+      +----------------+      | sort by CSV size |      | (if provided)    |
                                                +------------------+      +------------------+
                                                                                |
[2] MODE SPLIT <----------------------------------------------------------------+
       |
       +-----------------------------------+-----------------------------------+
       |                                   |                                   |
+---------------------+         +------------------+                +----------------------+
| --export-questions  |         |   --estimate     |                |      run mode        |
+---------------------+         +------------------+                +----------------------+
| write selected JSON |         | estimate tokens  |                | validate config      |
| -> exit             |         | + cost -> exit   |                | create runner        |
+---------------------+         +------------------+                +----------------------+
                                                                      |
[3] RESUME SPLIT (RUN MODE ONLY) <------------------------------------+
       |
       +-----------------------------------+-----------------------------------+
       |                                   |                                   |
+---------------------+         +------------------+                +----------------------+
| --resume-from-file  |         | --resume-batch   |                |     normal run       |
+---------------------+         +------------------+                +----------------------+
| parse local JSONL   |         | poll remote      |                | runner.run(...)      |
| merge checkpoint    |         | batch_id         |                +----------------------+
| -> exit             |         | merge -> exit    |                         |
+---------------------+         +------------------+                         v

[4] RUNNER TEMPLATE (BaseRunner.run)
+--------------------------------------------------------------------------------------+
| 1) load checkpoint: output/<checkpoint>/<model_id>.json                              |
| 2) compute done indices (respect --retry-errors)                                      |
| 3) filter remaining questions                                                         |
| 4) sort by db_id (maximize cache hits)                                                |
| 5) chunk into mini-batches                                                            |
| 6) for each batch: provider _process_batch -> extract_result -> build answer -> save |
+--------------------------------------------------------------------------------------+
                                         |
[5] PROVIDER EXECUTION                   v
+--------------------------------------------------------------------------------------+
| Claude     : Files API (or inline fallback) + code_execution + batch/async          |
| Gemini     : inline CSV bytes + code_execution + async                               |
| OpenAI     : upload file_ids + Responses API/code_interpreter + batch/async          |
| OpenRouter : embed full CSV text + chat completions (no code interpreter) + async    |
+--------------------------------------------------------------------------------------+
                                         |
[6] PARSING + OUTPUT                      v
+--------------------------------------------------------------------------------------+
| raw response -> parse/normalize JSON array-of-arrays                                 |
| parse fail  -> mark ERROR:...                                                         |
| append result item -> checkpoint saved after each mini-batch                          |
+--------------------------------------------------------------------------------------+
                                         |
[7] FINAL ARTIFACTS                       v
+---------------------------------------------------------------+
| file     : output/<checkpoint>/<model_id>.json               |
| terminal : progress logs + error count summary               |
+---------------------------------------------------------------+
```