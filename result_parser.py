"""
Extract the final JSON result from raw API responses.

Each provider embeds the code-execution output differently:
  - Claude  : bash_code_execution_tool_result block (stdout from code_execution tool)
  - Gemini  : Part with executable_code / code_execution_result
  - OpenAI  : output item of type "code_interpreter_call" (Responses API)

All parsers return a tuple (result_str, raw_output):
  - result_str : JSON string like `[["val1", 2]]`, or `"ERROR:..."` on failure
  - raw_output : raw stdout / text from code execution, before JSON parsing
                 (empty string if nothing was captured)
"""

import ast
import json
import re
from typing import Any


# Regex to find the last JSON array-of-arrays on a line
# Use [^\]]* instead of .*? to avoid catastrophic backtracking when the outer ] is missing
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*)?\]")

# Regex to detect a pandas-style table: lines with index + whitespace-separated cols
# e.g. "6  Renewable  1549.33"
_PANDAS_ROW_RE = re.compile(r"^\s*\d+\s+\S.*$", re.MULTILINE)

# Regex to find a Python list literal (greedy — use per-paragraph, not on full text)
_PY_LIST_RE = re.compile(r"\[[\s\S]*\]")

# Regex to replace numpy scalar constructors with their values: np.float64(x) → x
_NP_TYPE_RE = re.compile(r"\bnp\.\w+\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)")

# Column names from power_plants_vn that triggered a false positive when the model
# accidentally printed the column-selection list instead of the result:
# result[["std_fuel", "capacity_mw"]].values.tolist() → [["std_fuel", "capacity_mw"]]
_SCHEMA_COLUMNS = {
  "std_fuel", "capacity_mw", "std_name", "commissioning_year",
  "latitude", "longitude", "country", "owner", "source", "db_id",
}


def _is_column_name_artifact(parsed: list) -> bool:
  """Return True if the parsed list looks like a pandas column-selection false positive."""
  if len(parsed) != 1 or not isinstance(parsed[0], list):
    return False
  inner = parsed[0]
  return bool(inner) and all(isinstance(v, str) and v in _SCHEMA_COLUMNS for v in inner)


def _parse_python_list_candidate(candidate: str) -> list | None:
  """
  Parse a Python list literal candidate safely and return the parsed list.
  Returns None if parsing fails or parsed object isn't a list.
  """
  try:
    cleaned = re.sub(r"\bnan\b", "None", candidate)
    cleaned = _NP_TYPE_RE.sub(r"\1", cleaned)
    parsed = ast.literal_eval(cleaned)
    if isinstance(parsed, list):
      return parsed
  except (ValueError, SyntaxError):
    return None
  return None


# Phrases that indicate the model hit its tool-call limit and never actually ran the
# code — any "result" array found alongside these is fabricated column names / paths.
_TOOL_LIMIT_RE = re.compile(
  r"(?i)(?:"
  r"(?:run|ran) out of tool calls?"
  r"|out of tool calls?"
  r"|used up (?:all )?my tool calls?"
  r"|used all my tool calls?"
  r"|hit the tool(?:[ -]call)? limit"
  r"|reached the tool(?:[ -]call)? limit"
  r"|tool(?:[ -]call)? (?:limit|quota|exhausted|limit reached|rate.limit|unavailable)"
  r"|(?:all )?tool calls? (?:for this turn|this turn|exhausted|limit reached)"
  r"|too many tool calls?"
  r"|I(?:'ve| have) (?:hit|reached|exceeded) (?:the |a )?(?:tool|limit|max)"
  r"|I(?:'ve| have) (?:used|consumed|spent) (?:up )?(?:all|my) tool"
  r"|exhausted (?:my |the )?tool"
  r"|I apologize(?:.{0,60})(?:tool|limit|execution)"
  r"|apologize(?:.{0,30})execution environment"
  r"|persistent(?:ly)? (?:tool|execution) (?:error|issue|fail)"
  r"|please run (?:the|this|it) (?:code|script|query|above)"
  r"|run it in your environment"
  r"|temporar(?:ily|y) unavailable"
  r"|execution environment.*unavailable"
  r"|all tool calls? are returning"
  r"|rather than a guessed? answer"
  r"|moment the tool is back"
  r"|could you please retry"
  r")"
)


def _parse_pandas_table(text: str) -> str | None:
  """
  Try to parse the last pandas-formatted table in text.
  Returns JSON array-of-arrays or None if not detected.
  Only attempts if >=2 data rows found (avoids false positives).
  """
  # Find lines that look like pandas rows (leading integer index)
  rows = []
  for line in text.strip().splitlines():
    m = re.match(r"^\s*\d+\s+(.*\S)\s*$", line)
    if m:
      # Split by 2+ spaces to handle values with single spaces
      cols = re.split(r"\s{2,}", m.group(1).strip())
      rows.append(cols)

  if len(rows) < 2:
    return None

  # Try to coerce numeric strings to float/int
  def _coerce(v: str):
    try:
      f = float(v)
      return int(f) if f == int(f) else round(f, 6)
    except ValueError:
      return v

  return json.dumps([[_coerce(c) for c in row] for row in rows], ensure_ascii=False)


def _extract_json_from_text(text: str) -> str:
  """Find the last valid JSON array-of-arrays in `text`."""
  text = text.strip()
  if not text:
    return "ERROR:no_output"

  if _TOOL_LIMIT_RE.search(text):
    return "ERROR:tool_limit_exceeded"

  # Try to parse the whole text first (model might output only JSON)
  try:
    parsed = json.loads(text)
    if isinstance(parsed, list):
      return json.dumps(parsed, ensure_ascii=False)
  except json.JSONDecodeError:
    pass

  # Find all JSON-array candidates, pick the last one.
  # Cap to last 3000 chars to avoid catastrophic backtracking on huge truncated arrays
  # (json.loads above already handles the full-text case for valid arrays).
  search_text = text[-3000:] if len(text) > 3000 else text
  candidates = _JSON_ARRAY_RE.findall(search_text)
  for candidate in reversed(candidates):
    try:
      parsed = json.loads(candidate)
      if isinstance(parsed, list) and not _is_column_name_artifact(parsed):
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
      continue

  # Fallback: model may print Python list with single quotes e.g. [['a', 1], ['b', 2]]
  # Also handles nan (pandas NaN) → None and numpy types np.float64(x) → x.
  # Split by blank lines first so the greedy _PY_LIST_RE doesn't span multiple blocks.
  for paragraph in reversed(text.split("\n\n")):
    paragraph = paragraph.strip()
    if not paragraph:
      continue
    py_candidates = _PY_LIST_RE.findall(paragraph)
    for candidate in reversed(py_candidates):
      parsed = _parse_python_list_candidate(candidate)
      if parsed is not None and not _is_column_name_artifact(parsed):
        return json.dumps(parsed, ensure_ascii=False)

  # If no paragraph-level hit, try the whole text once. This recovers very long
  # single-block outputs where the list exceeds paragraph-size heuristics.
  full_text_candidate = text.strip()
  if full_text_candidate.startswith("[") and full_text_candidate.endswith("]"):
    parsed_full = _parse_python_list_candidate(full_text_candidate)
    if parsed_full is not None and not _is_column_name_artifact(parsed_full):
      return json.dumps(parsed_full, ensure_ascii=False)

  # Fallback: try to parse a pandas-style printed DataFrame
  # Use only the trailing portion after the last blank line (skip CSV preview noise)
  last_block = text.rsplit("\n\n", 1)[-1]
  pandas_result = _parse_pandas_table(last_block)
  if pandas_result is not None:
    return pandas_result

  return f"ERROR:parse_failed:{text[:200]!r}"


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

def extract_result_claude(response: Any) -> tuple[str, str]:
  """
  Scan response.content blocks for bash_code_execution_tool_result
  (stdout from the code_execution_20250825 tool).
  Falls back to scanning text blocks if none found.

  Returns (result_str, raw_output).
  raw_output format:
    [CODE]
    <python code the model wrote>
    [/CODE]
    [OUTPUT]
    <stdout from code execution>
    [/OUTPUT]
  """
  tool_output_parts: list[str] = []
  text_parts: list[str] = []
  code_parts: list[str] = []

  for block in response.content:
    btype = getattr(block, "type", None)

    if btype == "server_tool_use":
      # Capture the Python code the model submitted to code_execution
      inp = getattr(block, "input", None)
      code = None
      if isinstance(inp, dict):
        code = inp.get("code")
      elif inp is not None:
        code = getattr(inp, "code", None)
      if code:
        code_parts.append(str(code))

    elif btype == "bash_code_execution_tool_result":
      # content is a single BetaBashCodeExecutionResultBlock (or error block)
      # BetaBashCodeExecutionResultBlock has .stdout, .stderr, .return_code
      content = getattr(block, "content", None)
      if content is not None:
        sub_type = getattr(content, "type", None)
        if sub_type == "bash_code_execution_result":
          stdout = getattr(content, "stdout", "") or ""
          tool_output_parts.append(stdout)

    elif btype == "tool_result":
      # Backward-compat: old tool_result shape (shouldn't appear with
      # code_execution_20250825 but kept for safety)
      content = getattr(block, "content", None)
      if isinstance(content, list):
        for sub in content:
          if getattr(sub, "type", None) == "text":
            tool_output_parts.append(sub.text)
      elif isinstance(content, str):
        tool_output_parts.append(content)

    elif btype == "text":
      text_parts.append(getattr(block, "text", ""))

  stdout = "\n".join(tool_output_parts) if tool_output_parts else "\n".join(text_parts)

  # Build raw_output with code section for debugging
  if code_parts:
    code_section = "\n\n".join(code_parts)
    raw_output = f"[CODE]\n{code_section}\n[/CODE]\n\n[OUTPUT]\n{stdout}\n[/OUTPUT]"
  else:
    raw_output = stdout

  if not stdout.strip():
    return "ERROR:no_output", raw_output
  return _extract_json_from_text(stdout), raw_output


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def extract_result_gemini(response: Any) -> tuple[str, str]:
  """
  Scan response.candidates[0].content.parts for code_execution_result.
  Falls back to text parts.

  Returns (result_str, raw_output).
  """
  exec_output_parts: list[str] = []
  text_parts: list[str] = []

  try:
    parts = response.candidates[0].content.parts
  except (AttributeError, IndexError):
    return "ERROR:no_candidates", ""

  for part in parts:
    exec_result = getattr(part, "code_execution_result", None)
    if exec_result is not None:
      output = getattr(exec_result, "output", "") or ""
      exec_output_parts.append(output)
    else:
      text = getattr(part, "text", None)
      if text:
        text_parts.append(text)

  raw_output = "\n".join(exec_output_parts) if exec_output_parts else "\n".join(text_parts)
  if not raw_output.strip():
    return "ERROR:no_output", ""
  return _extract_json_from_text(raw_output), raw_output


# ---------------------------------------------------------------------------
# OpenAI (Responses API)
# ---------------------------------------------------------------------------

def extract_result_openai(response: Any) -> tuple[str, str]:
  """
  Scan response.output items for code_interpreter_call outputs.
  Falls back to message text.

  Returns (result_str, raw_output).
  """
  exec_output_parts: list[str] = []
  text_parts: list[str] = []

  output_items = getattr(response, "output", []) or []
  for item in output_items:
    item_type = getattr(item, "type", None)
    if item_type == "code_interpreter_call":
      outputs = getattr(item, "outputs", []) or []
      for out in outputs:
        if getattr(out, "type", None) == "logs":
          exec_output_parts.append(getattr(out, "logs", "") or "")
    elif item_type == "message":
      content = getattr(item, "content", []) or []
      for c in content:
        if getattr(c, "type", None) == "output_text":
          text_parts.append(getattr(c, "text", "") or "")

  raw_output = "\n".join(exec_output_parts) if exec_output_parts else "\n".join(text_parts)
  if not raw_output.strip():
    return "ERROR:no_output", ""
  return _extract_json_from_text(raw_output), raw_output


# ---------------------------------------------------------------------------
# OpenRouter (Chat Completions — assistant message text)
# ---------------------------------------------------------------------------

def _strip_markdown_code_fence(text: str) -> str:
  """If the model wrapped JSON in ``` / ```json, strip the fence."""
  t = text.strip()
  if not t.startswith("```"):
    return text
  lines = t.split("\n")
  if not lines:
    return text
  # drop opening ``` or ```json
  lines = lines[1:]
  if lines and lines[-1].strip() == "```":
    lines = lines[:-1]
  return "\n".join(lines)


def extract_result_openrouter(response: Any) -> tuple[str, str]:
  """
  Read choices[0].message.content from Chat Completions response.
  """
  try:
    choice = response.choices[0]
    content = choice.message.content
  except (AttributeError, IndexError, TypeError):
    return "ERROR:no_output", ""

  if content is None:
    return "ERROR:no_output", ""

  text = str(content)
  stripped = _strip_markdown_code_fence(text)
  raw_output = text
  return _extract_json_from_text(stripped), raw_output


# ---------------------------------------------------------------------------
# DeepSeek (tool-calling loop — local execution, multi-round)
# ---------------------------------------------------------------------------

def extract_result_deepseek(response: Any) -> tuple[str, str]:
  """
  Read from SimpleNamespace(code_parts, exec_outputs, final_text,
  submitted_answer, tool_limit_exceeded) returned by DeepSeekRunner._call_one_async().

  Priority order:
    1. submitted_answer — model called submit_answer tool (canonical path)
    2. tool_limit_exceeded — runner hit _MAX_TOOL_ROUNDS without submit → error
    3. final_text — model stopped without calling submit_answer (fallback)
    4. last exec_output only — convergence-guard or stuck-loop exit (last resort)

  Returns (result_str, raw_output).
  """
  code_parts: list[str] = getattr(response, "code_parts", []) or []
  exec_outputs: list[str] = getattr(response, "exec_outputs", []) or []
  final_text: str = getattr(response, "final_text", "") or ""
  submitted_answer = getattr(response, "submitted_answer", None)
  tool_limit_exceeded: bool = getattr(response, "tool_limit_exceeded", False)

  # raw_output shows all rounds for inspection; last exec_output is the final stdout.
  all_outputs = "\n\n".join(exec_outputs) if exec_outputs else final_text
  if code_parts:
    code_section = "\n\n---\n\n".join(code_parts)
    raw_output = f"[CODE]\n{code_section}\n[/CODE]\n\n[OUTPUT]\n{all_outputs}\n[/OUTPUT]"
  else:
    raw_output = all_outputs

  # Priority 1: model used submit_answer — trust it unconditionally.
  if submitted_answer is not None:
    try:
      return json.dumps(submitted_answer, ensure_ascii=False), raw_output
    except (TypeError, ValueError):
      return "ERROR:invalid_submitted_answer", raw_output

  # Priority 2: runner exhausted tool rounds without a submit.
  if tool_limit_exceeded:
    return "ERROR:tool_limit_exceeded", raw_output

  # Priority 3: model stopped and may have put the answer in its final message.
  if final_text.strip():
    return _extract_json_from_text(final_text), raw_output

  # Priority 4: convergence-guard or other early exit — only inspect the last
  # exec_output, not the whole concat (avoids picking up exploration prints).
  if exec_outputs:
    last = exec_outputs[-1]
    if last.strip():
      return _extract_json_from_text(last), raw_output

  return "ERROR:no_output", raw_output


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS = {
  "claude": extract_result_claude,
  "gemini": extract_result_gemini,
  "openai": extract_result_openai,
  "openrouter": extract_result_openrouter,
  "deepseek": extract_result_deepseek,
}


def extract_result(response: Any, provider: str) -> tuple[str, str]:
  """
  Returns (result_str, raw_output).
    result_str : parsed JSON string or "ERROR:..."
    raw_output : raw code-execution stdout, for debugging / re-parsing
  """
  extractor = _EXTRACTORS.get(provider)
  if extractor is None:
    raise ValueError(f"Unknown provider: {provider!r}")
  try:
    return extractor(response)
  except Exception as exc:  # noqa: BLE001
    return f"ERROR:exception:{type(exc).__name__}:{exc!s:.200}", ""
