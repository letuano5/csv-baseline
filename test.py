"""
test.py — Chứng minh DeepSeek endpoint không hỗ trợ Files API (upload file).

Dùng OpenAI SDK với base_url của DeepSeek, thử gọi client.files.create()
giống hệt như openai_runner.py làm — và in ra lỗi nhận được.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

CSV_PATH = Path(__file__).parent / "input" / "csv" / "bank_customer_churn.csv"

print(f"Base URL : {DEEPSEEK_BASE_URL}")
print(f"File     : {CSV_PATH.name} ({CSV_PATH.stat().st_size:,} bytes)")
print()

client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)

print("Đang thử upload file qua Files API (client.files.create) ...")
try:
  with open(CSV_PATH, "rb") as fh:
    file_obj = client.files.create(
      file=(CSV_PATH.name, fh, "text/csv"),
      purpose="user_data",
    )
  print(f"Upload thành công! file_id = {file_obj.id}")
except Exception as exc:
  print(f"THẤT BẠI — {type(exc).__name__}")
  print(f"Chi tiết: {exc}")
  status = getattr(getattr(exc, "response", None), "status_code", None)
  if status:
    print(f"HTTP status: {status}")
