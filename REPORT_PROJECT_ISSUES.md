# Báo Cáo Lỗi Và Cách Xử Lý (Timeline)

Tài liệu này tổng hợp các lỗi/chồng chéo đã gặp trong project từ giai đoạn chuẩn hóa CSV đến pipeline đánh giá kết quả.

## 1) Giai đoạn chuẩn hóa CSV (`normalize_csvs.py`)

- Vấn đề:
  - CSV không đồng nhất encoding và delimiter (BOM, `;`, tab...).
  - Header bẩn (khoảng trắng, ký tự tiếng Việt, tên cột không ổn định).
  - Missing values không đồng nhất (`N/A`, `-`, `null`, `none`...).
  - Định dạng ngày/số lộn xộn (dd/mm/yyyy, yyyy-mm-dd, dấu phẩy thập phân, dấu phẩy ngăn cách hàng nghìn...).
- Cách xử lý:
  - Chuẩn hóa output về `utf-8` (không BOM) và delimiter `,`.
  - Làm sạch header + slugify về ASCII snake_case, có cơ chế tránh trùng tên cột.
  - Chuẩn hóa missing markers về chuỗi rỗng.
  - Parse và chuẩn hóa ngày về `YYYY-MM-DD` nếu nhận diện được.
  - Chuẩn hóa numeric-like string theo bộ luật dấu phẩy/chấm/currency.
- Tác động:
  - Giảm lỗi parse/SQL do format.
  - Tạo nền data ổn định cho các bước prompt, runner, evaluator.

## 2) Xử lý trường hợp đặc thù dataset cơ sở y tế

- Vấn đề:
  - Cột `ngay_cong_bo` có khi chứa kèm note trong ngoặc, nhất là thông tin "SYT nhận ...", gây khó parse và khó query.
- Cách xử lý:
  - Tách thông tin thành các cột rõ nghĩa:
    - `ngay_cong_bo`
    - `ngay_syt_nhan`
    - `ghi_chu_ngay_cong_bo`
  - Bổ sung logic hậu xử lý dataset `facility_announcements_2025.csv`.
- Tác động:
  - Query theo ngày đúng hơn.
  - Giữ được dữ liệu nguyên gốc và tăng khả năng kiểm tra.

## 3) Sinh profile dữ liệu tự động (`generate_profiles.py`)

- Vấn đề:
  - Model thiếu ngữ cảnh về schema/chất lượng cột nên dễ nhầm type.
  - Khó phát hiện cột số dạng text, cột ngày nhiều format, cột thiếu dữ liệu cao.
- Cách xử lý:
  - Sinh profile cho mỗi CSV (row_count, column_count, inferred_types, quality_flags, column_stats).
  - Tính các chỉ số: missing_ratio, unique_ratio, top_values, sample_values, date/numeric/bool match ratio.
  - Đánh dấu quality flags (BOM, delimiter lạ, mixed date format).
- Tác động:
  - Prompt có context tốt hơn.
  - Giảm sai do hiểu nhầm dữ liệu.

## 4) Metadata CSV hardcode dễ bị lệch (`config.py`)

- Vấn đề:
  - Khi đổi file CSV, metadata hardcode dễ sai encoding/delimiter/header.
- Cách xử lý:
  - Chuyển sang lazy metadata từ file thật:
    - auto detect encoding/delimiter
    - đọc header trực tiếp
    - cache bằng `lru_cache`
  - Giữ tương thích callsite qua `CSV_REGISTRY` proxy.
- Tác động:
  - Giảm "schema drift".
  - Dễ thay data mới mà không sửa tay metadata.

## 5) OpenRouter mode mismatch (code vs JSON kết quả)

- Vấn đề:
  - OpenRouter có lúc trả về Python code thay vì JSON array final.
  - Parser nhận output không đúng định dạng -> `ERROR`/invalid result.
- Cách xử lý:
  - Thêm logic:
    - detect/extract Python code (có/không có code fence),
    - chạy local trong temp dir với `data.csv`,
    - lấy `stdout` làm output để đưa vào parser/evaluator.
- Tác động:
  - Pipeline bên OpenRouter ổn định hơn.
  - Giảm lỗi do sai mode phản hồi.

## 6) Hợp đồng output của model chưa đủ chặt (`prompt.py`)

- Vấn đề:
  - Model in thêm text/fence hoặc nhiều `print`.
  - Không đảm bảo đúng thứ tự cột theo câu hỏi.
  - Số thực quá nhiều chữ số gây sai lệch khi so sánh.
- Cách xử lý:
  - Siết chặt output contract:
    - bắt buộc 1 `print` duy nhất,
    - in JSON array-of-arrays "sạch",
    - dùng `json.dumps(..., ensure_ascii=False)`,
    - nhắc rõ thứ tự cột phải trùng yêu cầu.
- Tác động:
  - Giảm lỗi parse và lỗi "format đúng logic sai".

## 7) Sai thứ tự cột dù giá trị đúng (`runners/base.py` + `evaluator.py`)

- Vấn đề:
  - Kết quả có thể đúng giá trị nhưng đảo thứ tự cột -> bị chấm sai.
- Cách xử lý:
  - Thêm detect `column_order_mismatch` trong evaluator.
  - Nếu phát hiện trường hợp này, runner tự retry 1 lần với correction hint tập trung vào thứ tự cột.
- Tác động:
  - Cứu được nhiều case fail "kỹ thuật output", không phải fail "logic câu hỏi".

## 8) Đánh giá kết quả sau khi chạy chưa đầy đủ (`main.py` + `evaluator.py`)

- Vấn đề:
  - Trước đây chủ yếu thấy tổng lỗi, khó biết sai câu nào/sai kiểu gì.
- Cách xử lý:
  - Tích hợp evaluator sau mỗi run:
    - chạy SQL ground truth,
    - so sánh kết quả với rule số thực 2 chữ số,
    - xử lý NULL/NaN thống nhất,
    - in `wrong_indices` + preview mismatch,
    - lưu report JSON chi tiết.
- Tác động:
  - Vòng lặp debug nhanh hơn.
  - Theo dõi quality rõ ràng hơn qua từng lần chạy.

## 9) Các bài học rút ra

- Chuẩn hóa data vào là bước bắt buộc, không nên trì hoãn.
- Prompt phải có "contract output" rõ ràng và có test fail-fast.
- Eval cần tách bài bản:
  - value mismatch,
  - format/parse mismatch,
  - column-order mismatch,
  - sql execution mismatch.
- Nên có báo cáo chi tiết theo run (`.report.json`) để so sánh xu hướng.

## 10) Đề xuất tiếp theo (optional)

- Thêm script tổng hợp multi-run để so sánh accuracy qua checkpoint.
- Thêm regression set nhỏ (20-50 câu) để check nhanh trước run lớn.
- Ghi lại "known problematic db_id/questions" để retry có mục tiêu.

