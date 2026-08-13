## ADDED Requirements

### Requirement: MLflow tracking server chạy trong compose
Hệ thống SHALL cung cấp MLflow tracking server như một service trong `docker-compose.yml`, lắng nghe cổng 5001, dùng named volume để lưu backend store và artifact store.

#### Scenario: Truy cập MLflow UI
- **WHEN** chạy `docker compose up` và mở `http://localhost:5001`
- **THEN** MLflow UI hiển thị và liệt kê được các experiment

#### Scenario: Dữ liệu tồn tại qua restart
- **WHEN** restart MLflow container
- **THEN** mọi run đã ghi trước đó vẫn còn

### Requirement: Training baseline log lên MLflow
Training baseline SHALL log params, metrics và artifacts lên MLflow ở mỗi lần chạy.

#### Scenario: Chạy training baseline
- **WHEN** chạy `python -m aegis.models.train_baseline`
- **THEN** một run mới xuất hiện trong MLflow với params (`C`, `max_features`, `ngram_range`), metrics (`val_macro_f1`, `test_macro_f1`, F1 từng lớp) và artifacts (model, vectorizer, confusion matrix)

#### Scenario: Nhiều experiment được ghi lại
- **WHEN** grid `C` quét qua 7 giá trị
- **THEN** MLflow chứa ít nhất 7 run tương ứng, mỗi run có giá trị `C` riêng

### Requirement: Backfill run RoBERTa từ checkpoint
Hệ thống SHALL cung cấp script tạo lại run MLflow cho RoBERTa từ `roberta_checkpoints/*/trainer_state.json`, chỉ dùng dữ liệu đã ghi thật, MUST NOT bịa số liệu.

#### Scenario: Backfill thành công
- **WHEN** chạy `python scripts/backfill_mlflow.py`
- **THEN** MLflow chứa run RoBERTa với learning rate, batch size, số epoch và macro-F1 từng epoch khớp với `trainer_state.json`

#### Scenario: Đánh dấu nguồn gốc
- **WHEN** xem run được backfill
- **THEN** run có tag `source=backfill` và tag trỏ tới đường dẫn checkpoint gốc

### Requirement: Model Registry với stage
Hệ thống SHALL đăng ký cả hai model vào MLflow Model Registry với tên `aegis-baseline` và `aegis-roberta`, và MUST gán stage cho version đang phục vụ.

#### Scenario: Đăng ký model
- **WHEN** training hoàn tất và metric vượt ngưỡng
- **THEN** model được đăng ký và version mới nhất chuyển sang stage `Production`

#### Scenario: API nạp model từ registry
- **WHEN** API khởi động với `MODEL_SOURCE=registry`
- **THEN** API nạp model qua URI `models:/aegis-roberta/Production` và ghi log version đã nạp

### Requirement: Ghi lại kết quả OOD calibration
Kết quả recalibration OOD SHALL được log lên MLflow như metrics và artifacts.

#### Scenario: Log calibration run
- **WHEN** chạy `python scripts/recalibrate_ood.py`
- **THEN** MLflow có run chứa `auroc`, `target_fpr`, `measured_fpr`, `measured_recall` cho từng method, kèm artifact `ood_operating_points.json` và biểu đồ ROC
