## ADDED Requirements

### Requirement: OODDetector là hàm thuần trên logits
Interface `OODDetector` SHALL nhận đầu vào là mảng logits và trả về điểm bất thường. Detector MUST NOT tokenize, MUST NOT nạp model, MUST NOT đọc file trong quá trình chấm điểm.

#### Scenario: Chấm điểm bằng logits giả
- **WHEN** gọi `detector.score(np.array([[2.1, -0.3, 0.5, -1.2]]))`
- **THEN** trả về một số thực, không phát sinh lời gọi nạp model hay đọc file

#### Scenario: Mọi implementation tuân thủ cùng interface
- **WHEN** chạy bộ test interface parametrize qua `MSPDetector`, `EnergyDetector`, `EntropyDetector`, `NullOODDetector`
- **THEN** mọi implementation đều có `score`, `is_ood`, `method`, `enabled` và vượt qua toàn bộ assertion chung

### Requirement: MSP scoring
`MSPDetector` SHALL tính điểm bất thường bằng `1 - max(softmax(logits))`.

#### Scenario: Logits rất tự tin
- **WHEN** logits là `[[10.0, 0.0, 0.0, 0.0]]`
- **THEN** điểm bất thường gần 0

#### Scenario: Logits phân vân đều
- **WHEN** logits là `[[0.0, 0.0, 0.0, 0.0]]`
- **THEN** điểm bất thường bằng 0.75 với sai số 1e-6

### Requirement: Energy scoring
`EnergyDetector` SHALL tính điểm bằng `-T * logsumexp(logits / T)` với `T` đọc từ cấu hình.

#### Scenario: Công thức đúng
- **WHEN** logits là `[[1.0, 2.0, 3.0, 4.0]]` và `T = 1.0`
- **THEN** kết quả bằng `-logsumexp([1,2,3,4])` với sai số 1e-6

#### Scenario: Đơn điệu theo độ tự tin
- **WHEN** so sánh logits `[[10,0,0,0]]` với `[[1,0,0,0]]`
- **THEN** logits tự tin hơn cho điểm energy thấp hơn

### Requirement: Entropy scoring cho baseline
`EntropyDetector` SHALL tính entropy chuẩn hoá trên xác suất đầu ra của baseline, giá trị nằm trong [0, 1].

#### Scenario: Xác suất chắc chắn
- **WHEN** xác suất là `[1.0, 0.0, 0.0, 0.0]`
- **THEN** entropy chuẩn hoá bằng 0.0

#### Scenario: Xác suất đều
- **WHEN** xác suất là `[0.25, 0.25, 0.25, 0.25]`
- **THEN** entropy chuẩn hoá bằng 1.0

### Requirement: Ngưỡng đọc từ cấu hình, không hardcode
Detector SHALL đọc ngưỡng, nhiệt độ `T`, `max_len` và danh sách tên nhãn từ `ood_config.json`. Không giá trị nào trong số này được hardcode trong source.

#### Scenario: Tải cấu hình
- **WHEN** detector khởi tạo
- **THEN** ngưỡng và `T` khớp giá trị trong `ood_config.json`

#### Scenario: max_len khớp lúc train
- **WHEN** đọc `max_len` từ cấu hình
- **THEN** giá trị bằng 128, đúng như lúc calibrate

#### Scenario: Thiếu file cấu hình
- **WHEN** không tìm thấy `ood_config.json`
- **THEN** hệ thống dùng `NullOODDetector`, ghi log cảnh báo, và API vẫn phục vụ được phần phân loại

### Requirement: Recalibration sinh operating-point table
Script recalibration SHALL quét các mức FPR mục tiêu {1, 2, 5, 10, 15, 20, 30}% cho cả MSP và Energy, xuất `ood_operating_points.json` chứa ngưỡng, recall và AUROC ứng với từng mức.

#### Scenario: Sinh bảng đầy đủ
- **WHEN** chạy `python scripts/recalibrate_ood.py`
- **THEN** `ood_operating_points.json` chứa 14 dòng (7 mức FPR × 2 method), mỗi dòng có `threshold`, `measured_fpr`, `measured_recall`, `auroc`

#### Scenario: Chọn operating point mặc định
- **WHEN** recalibration hoàn tất với mục tiêu mặc định
- **THEN** `ood_config.json` mới có `target_fpr = 0.05` và `measured_fpr` không vượt quá 0.05

#### Scenario: Cache logits
- **WHEN** chạy recalibration lần thứ hai
- **THEN** script nạp logits từ file `.npy` đã cache thay vì chạy lại inference

### Requirement: Recalibration dùng đúng preprocessing lúc train
Recalibration SHALL đưa text thô vào RoBERTa với `max_len=128`. MUST NOT áp `clean_text_tfidf()` lên nhánh RoBERTa.

#### Scenario: Không có bước làm sạch trên nhánh RoBERTa
- **WHEN** thu thập logits để calibrate
- **THEN** text đưa vào tokenizer giống hệt text gốc của dataset

### Requirement: Bật tắt OOD qua feature flag
Hệ thống SHALL hỗ trợ biến môi trường `OOD_ENABLED`. Khi tắt, API vẫn phục vụ phân loại bình thường và trường `ood` trong response bằng `null`.

#### Scenario: OOD tắt
- **WHEN** `OOD_ENABLED=false` và gọi `/v1/predict`
- **THEN** response trả HTTP 200, có `predicted_class` và `confidence`, `ood` bằng `null`

#### Scenario: OOD bật
- **WHEN** `OOD_ENABLED=true` và gọi `/v1/predict`
- **THEN** `ood` chứa `is_ood`, `score`, `method`, `threshold`

#### Scenario: Trạng thái được phơi ra ngoài
- **WHEN** gọi `/v1/model/info`
- **THEN** response chứa `ood_enabled` phản ánh đúng cấu hình hiện tại
