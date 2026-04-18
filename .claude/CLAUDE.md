# Python Coding Conventions

## Version
Python 3.12

## Indentation

Sử dụng **2 spaces** cho mỗi mức indent (không dùng tab).

```python
def hello():
  if True:
    print("hello")
```

## Package Manager & Virtual Environment

Luôn dùng `uv` — không dùng `pip` hay `python` trực tiếp.

### Khởi tạo project

```bash
uv init
uv venv
```

### Thêm / xóa dependencies

```bash
uv add <package>      # thêm và tự cập nhật pyproject.toml + uv.lock
uv remove <package>   # xóa
```

`uv` tự lo `pyproject.toml` và `uv.lock` — không cần `requirements.txt`, không cần `pip freeze`.

## Chạy code

Luôn dùng `uv run` — không gọi `python` trực tiếp.

```bash
uv run main.py
```

`uv run` tự đảm bảo đúng venv và đủ dependencies trước khi chạy.

## Checklist trước khi chạy

- [ ] `uv init` + `uv venv` đã chạy
- [ ] Dependencies đã thêm qua `uv add`, không cài thủ công
- [ ] Chạy bằng `uv run <file>.py`, không dùng `python <file>.py`

## Environment Variables

Luôn lưu config và secrets vào file `.env`, không hardcode trong code.

```bash
# .env
DATABASE_URL=postgres://...
API_KEY=sk-...
DEBUG=true
```

Đọc trong code bằng `python-dotenv`:

```bash
uv add python-dotenv
```

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
```

> **Lưu ý:** Thêm `.env` vào `.gitignore` — không commit secrets lên repo.