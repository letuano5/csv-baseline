# CSV Baseline

Framework đánh giá khả năng trả lời câu hỏi tiếng Việt trên dữ liệu CSV của các mô hình ngôn ngữ lớn (Claude, GPT, Gemini, OpenRouter), sử dụng code execution phía máy chủ để tính kết quả chính xác thay vì sinh câu trả lời trực tiếp.

## Yêu cầu môi trường

- Python 3.12
- uv

## Cài đặt

```bash
uv sync
```

Tạo file `.env` từ mẫu và điền API key:

```bash
cp .env.example .env
```

Các biến cần thiết trong `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-v1-...
```

Hỗ trợ nhiều key cho mỗi provider (rotation khi bị rate limit):

```
ANTHROPIC_API_KEY_1=sk-ant-...
ANTHROPIC_API_KEY_2=sk-ant-...
```

## Chạy full luồng

### Bước 0 (tuỳ chọn): Chuẩn hóa và sinh profile CSV

Chuẩn hóa encoding, delimiter, header và giá trị thiếu trong các file CSV đầu vào:

```bash
uv run normalize_csvs.py --inplace
```

Sinh file profile mô tả kiểu dữ liệu từng cột (dùng để cải thiện prompt):

```bash
uv run generate_profiles.py
# Chỉ một dataset:
uv run generate_profiles.py --db-id danang_hsg_list
```

Profile được lưu tại `profiles/auto/<db_id>.json`. Nếu không chạy bước này, prompt vẫn hoạt động bình thường nhưng không có gợi ý về kiểu cột.

### Bước 1: Chia dataset (tuỳ chọn)

Xem có bao nhiêu câu hỏi và xuất tập con:

```bash
# Xuất 100 câu đầu (đã sắp xếp theo kích thước CSV tăng dần) ra file riêng
uv run main.py --limit 100 --export-questions input/questions_random_100.json
```

Tham số `--questions` cho phép chỉ định file câu hỏi khác mặc định (`input/questions.json`):

```bash
uv run main.py --questions input/questions_random_100.json --provider claude ...
```

### Bước 2: Ước tính chi phí

Trước khi chạy thực, kiểm tra ước tính token và chi phí cho từng mô hình:

```bash
# Tất cả mô hình mặc định
uv run main.py --estimate

# Một provider cụ thể
uv run main.py --provider claude --model-id claude-sonnet-4-6 --estimate

# Kết hợp với giới hạn câu hỏi
uv run main.py --provider openai --model-id gpt-5.4 --limit 200 --estimate
```

### Bước 3: Chạy đánh giá

Mỗi lần chạy cần chỉ định `--provider`, `--model-id` và `--checkpoint` (tên thư mục lưu kết quả):

```bash
# Claude
uv run main.py --provider claude --model-id claude-sonnet-4-6 --checkpoint run-01

# Gemini
uv run main.py --provider gemini --model-id gemini-2.5-pro-preview --checkpoint run-01

# OpenAI
uv run main.py --provider openai --model-id gpt-5.4 --checkpoint run-01

# OpenRouter
uv run main.py --provider openrouter --model-id google/gemma-4-26b-a4b-it --checkpoint run-01
```

Kết quả được lưu dạng JSON tại `output/<checkpoint>/<model_id>.json`. Tiến trình được checkpoint sau mỗi mini-batch; ngắt ngang và chạy lại lệnh trên sẽ tiếp tục từ câu hỏi còn dang dở.

#### Tham số bổ sung

| Tham số | Mô tả |
|---|---|
| `--limit N` | Chỉ xử lý N câu đầu tiên (smoke test) |
| `--retry-errors` | Chạy lại các câu đã có kết quả `ERROR:...` |
| `--resume-batch BATCH_ID` | Lấy kết quả từ một batch đã submit (Claude: `msgbatch_01...`, OpenAI: `batch_...`) |
| `--resume-from-file PATH` | Đọc file JSONL batch output đã tải về và merge vào checkpoint (chỉ OpenAI) |

### Bước 4: Re-parse kết quả

Nếu cải thiện logic parse mà không muốn gọi lại API, chạy lại parser trên file output hiện có:

```bash
# Tạo file mới .reparsed.json
uv run reparse.py output/run-01/claude-sonnet-4-6.json

# Ghi đè file gốc
uv run reparse.py output/run-01/claude-sonnet-4-6.json --inplace
```

## Cấu trúc output

Mỗi entry trong file JSON output có dạng:

```json
{
  "index": 0,
  "db_id": "stock_prices",
  "question": "Cổ phiếu nào có giá đóng cửa cao nhất?",
  "result": "[[\"VIC\", 85000]]",
  "raw_output": "[CODE]\n...\n[/CODE]\n\n[OUTPUT]\n...\n[/OUTPUT]"
}
```

Trường `result` là JSON array-of-arrays hoặc chuỗi `ERROR:...` nếu parse thất bại. Trường `raw_output` lưu stdout thô từ code execution, dùng để debug hoặc re-parse.

## Luồng xử lý

```
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
| 2) compute done indices (respect --retry-errors)                                     |
| 3) filter remaining questions                                                        |
| 4) sort by db_id (maximize cache hits)                                               |
| 5) chunk into mini-batches                                                           |
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
[6] PARSING + OUTPUT                     v
+--------------------------------------------------------------------------------------+
| raw response -> parse/normalize JSON array-of-arrays                                 |
| parse fail  -> mark ERROR:...                                                        |
| append result item -> checkpoint saved after each mini-batch                         |
+--------------------------------------------------------------------------------------+
                                         |
[7] FINAL ARTIFACTS                      v
+---------------------------------------------------------------+
| file     : output/<checkpoint>/<model_id>.json               |
| terminal : progress logs + error count summary               |
+---------------------------------------------------------------+
```
