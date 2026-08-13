## ADDED Requirements

### Requirement: Data ingestion tái lập được
Hệ thống SHALL tải AG News qua thư viện `datasets`, cache xuống đĩa local, và MUST hoạt động được ở lần chạy thứ hai mà không cần mạng.

#### Scenario: Lần tải đầu tiên
- **WHEN** chạy `python -m aegis.data.loader` lần đầu
- **THEN** dataset được tải về cache local và ghi log số dòng train/test

#### Scenario: Chạy lại khi không có mạng
- **WHEN** chạy lại loader với cache đã tồn tại và không có kết nối mạng
- **THEN** loader đọc từ cache và hoàn tất không lỗi

### Requirement: Dataset versioning
Hệ thống SHALL tính hash SHA256 của dataset cùng số dòng và phân bố lớp, ghi vào `data/dataset_card.json`.

#### Scenario: Sinh dataset card
- **WHEN** data pipeline chạy xong
- **THEN** `data/dataset_card.json` tồn tại và chứa `sha256`, `n_train`, `n_test`, `class_counts`

#### Scenario: Phát hiện dataset đổi
- **WHEN** hash tính được khác hash đã ghi trong dataset card
- **THEN** pipeline ghi cảnh báo rõ ràng nêu cả hai giá trị hash

### Requirement: Preprocessing tách theo nhánh model
Hệ thống SHALL cung cấp hai hàm tiền xử lý riêng biệt. `clean_text_tfidf()` MUST chỉ được dùng cho nhánh TF-IDF. Nhánh RoBERTa MUST nhận text thô, không qua bất kỳ bước làm sạch nào.

#### Scenario: Nhánh TF-IDF làm sạch text
- **WHEN** gọi `clean_text_tfidf("Apple's Q3 revenue hit $89.5B in 2024!")`
- **THEN** kết quả là chữ thường, không còn chữ số, không còn ký tự đặc biệt

#### Scenario: Nhánh RoBERTa giữ nguyên text
- **WHEN** RobertaPredictor xử lý `"Apple's Q3 revenue hit $89.5B in 2024!"`
- **THEN** chuỗi đưa vào tokenizer giống hệt chuỗi đầu vào, giữ nguyên hoa thường, chữ số và dấu câu

### Requirement: Train/validation/test split tái lập chính xác
Hệ thống SHALL chia train thành 90/10 stratify theo label với `random_state=42`, và MUST ghi index ra file để tái lập được.

#### Scenario: Split ổn định qua nhiều lần chạy
- **WHEN** chạy split hai lần với cùng seed
- **THEN** index của train và validation giống hệt nhau ở cả hai lần

#### Scenario: Giữ nguyên tỷ lệ lớp
- **WHEN** split hoàn tất
- **THEN** mỗi lớp chiếm 24%–26% ở cả train lẫn validation

### Requirement: Baseline dùng LogisticRegression có xác suất
Baseline SHALL dùng `LogisticRegression` trên đặc trưng TF-IDF (`max_features=50000`, `ngram_range=(1,2)`, `sublinear_tf=True`, `stop_words="english"`) và MUST hỗ trợ `predict_proba`.

#### Scenario: Baseline trả xác suất hợp lệ
- **WHEN** gọi `predict_proba` trên một văn bản bất kỳ
- **THEN** trả về mảng 4 phần tử, mỗi phần tử trong [0, 1], tổng bằng 1.0 (sai số 1e-6)

#### Scenario: Tuning siêu tham số
- **WHEN** chạy training baseline
- **THEN** grid `C` được quét bằng cross-validation và giá trị `C` tốt nhất được ghi lại

### Requirement: Model comparison trên cùng test set
Hệ thống SHALL đánh giá baseline và RoBERTa trên cùng test set với cùng bộ metric, xuất `model_comparison.json`.

#### Scenario: Sinh bảng so sánh
- **WHEN** chạy `python -m aegis.models.compare`
- **THEN** `model_comparison.json` chứa `val_macro_f1` và `test_macro_f1` cho cả hai model

### Requirement: Evaluation sinh artifact đầy đủ
Hệ thống SHALL xuất classification report, confusion matrix và macro-F1 cho mỗi model được đánh giá.

#### Scenario: Đánh giá RoBERTa
- **WHEN** chạy evaluation trên RoBERTa
- **THEN** sinh ra classification report dạng JSON, ảnh confusion matrix, và macro-F1 ≥ 0.90

### Requirement: Dependency pin cứng
`requirements.txt` SHALL pin chính xác version của mọi dependency trực tiếp, bao gồm `transformers` khớp với `transformers_version` ghi trong `roberta_final/config.json`.

#### Scenario: Kiểm tra pin
- **WHEN** đọc `requirements.txt`
- **THEN** mọi dòng dùng toán tử `==`, không có dòng nào dùng `>=` hoặc để trống version
