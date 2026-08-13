## ADDED Requirements

### Requirement: Đủ bốn loại test theo rubric
Bộ test SHALL bao gồm unit test, integration test, data quality test và model validation test, tách theo thư mục và đánh dấu bằng pytest marker.

#### Scenario: Chạy theo loại
- **WHEN** chạy `pytest -m unit`, `-m integration`, `-m data`, `-m model`
- **THEN** mỗi lệnh chọn đúng tập test tương ứng và có ít nhất một test được thu thập

### Requirement: Ngưỡng độ phủ 80%
Cấu hình test SHALL bắt buộc độ phủ tối thiểu 80% và fail khi không đạt.

#### Scenario: Không đạt độ phủ
- **WHEN** độ phủ tụt dưới 80%
- **THEN** lệnh pytest kết thúc với exit code khác 0

### Requirement: Unit test không nạp model thật
Toàn bộ unit test SHALL chạy mà không nạp bất kỳ model thật nào và không cần mạng.

#### Scenario: Chạy offline
- **WHEN** chạy `pytest -m unit` khi không có mạng và chưa có model trên đĩa
- **THEN** toàn bộ test vượt qua

### Requirement: Test khoá train/serve skew
Bộ test SHALL khẳng định nhánh RoBERTa không áp preprocessing của TF-IDF.

#### Scenario: Nhánh RoBERTa nhận text thô
- **WHEN** RobertaPredictor xử lý một văn bản có chữ hoa, chữ số và dấu câu
- **THEN** chuỗi đưa vào tokenizer giống hệt chuỗi gốc

#### Scenario: Nhánh baseline có làm sạch
- **WHEN** BaselinePredictor xử lý cùng văn bản đó
- **THEN** chuỗi đưa vào vectorizer đã được làm sạch

### Requirement: Test hợp đồng ánh xạ nhãn
Bộ test SHALL khẳng định ánh xạ chỉ số sang tên nhãn đúng và không bao giờ lộ nhãn dạng `LABEL_x`.

#### Scenario: Ánh xạ đúng thứ tự
- **WHEN** ánh xạ các chỉ số 0, 1, 2, 3
- **THEN** kết quả lần lượt là `World`, `Sports`, `Business`, `Sci/Tech`

#### Scenario: Không lộ nhãn thô
- **WHEN** kiểm tra mọi giá trị nhãn API có thể trả về
- **THEN** không giá trị nào khớp mẫu `LABEL_\d`

### Requirement: Test hợp đồng max_len
Bộ test SHALL khẳng định độ dài chuỗi tối đa dùng khi phục vụ khớp với lúc calibrate.

#### Scenario: Giá trị max_len
- **WHEN** đọc `max_len` từ cấu hình đang chạy
- **THEN** giá trị bằng 128

### Requirement: Test giao diện OOD parametrize qua mọi implementation
Bộ test SHALL chạy cùng một tập assertion trên mọi implementation của `OODDetector`, kể cả `NullOODDetector`.

#### Scenario: Thêm implementation mới
- **WHEN** một implementation mới được đăng ký nhưng không tuân thủ interface
- **THEN** test giao diện fail

### Requirement: Integration test phủ toàn bộ endpoint
Integration test SHALL gọi mọi endpoint qua test client với predictor được mock.

#### Scenario: Đường thành công
- **WHEN** chạy integration test cho `/v1/predict`, `/v1/predict/batch`, `/v1/model/info`, `/health`, `/ready`, `/metrics`
- **THEN** mọi endpoint trả về mã trạng thái và cấu trúc body đúng như spec

#### Scenario: Đường lỗi
- **WHEN** gửi payload không hợp lệ tới mỗi endpoint
- **THEN** trả về mã lỗi đúng kèm body lỗi đúng cấu trúc

#### Scenario: Metrics phản ánh lưu lượng
- **WHEN** gọi `/v1/predict` ba lần rồi đọc `/metrics`
- **THEN** counter tương ứng tăng đúng 3

### Requirement: Data quality test
Bộ test SHALL kiểm tra schema, miền giá trị nhãn, cân bằng lớp, kích thước tập, giá trị rỗng, khoảng độ dài văn bản, trùng lặp, rò rỉ giữa train và test, và tính tái lập của split.

#### Scenario: Miền giá trị nhãn
- **WHEN** kiểm tra cột label
- **THEN** mọi giá trị thuộc {0, 1, 2, 3}

#### Scenario: Không rò rỉ dữ liệu
- **WHEN** so giao giữa tập text của train và test
- **THEN** phần giao rỗng

#### Scenario: Không có văn bản rỗng
- **WHEN** kiểm tra cột text
- **THEN** không dòng nào rỗng hoặc chỉ chứa khoảng trắng

### Requirement: Model validation test có ngưỡng số cụ thể
Bộ test SHALL kiểm chứng macro-F1 tổng thể, F1 từng lớp, recall và FPR của OOD, ngân sách độ trễ, tính bất biến, tính định hướng, tính tất định và hồi quy so với dự đoán vàng.

#### Scenario: Ngưỡng macro-F1
- **WHEN** đánh giá model đang phục vụ trên mẫu test
- **THEN** macro-F1 không thấp hơn 0.90

#### Scenario: Sàn F1 từng lớp
- **WHEN** kiểm tra F1 của từng lớp
- **THEN** mọi lớp đạt ít nhất 0.85

#### Scenario: Trần FPR của OOD
- **WHEN** đánh giá OOD sau recalibration
- **THEN** FPR không vượt quá 0.10

#### Scenario: Ngân sách độ trễ
- **WHEN** đo độ trễ một lời gọi dự đoán trên CPU
- **THEN** p95 dưới 500ms

#### Scenario: Bất biến với khoảng trắng
- **WHEN** thêm khoảng trắng thừa vào đầu và cuối văn bản
- **THEN** nhãn dự đoán không đổi

#### Scenario: Định hướng đúng trên tập vàng
- **WHEN** dự đoán trên tập câu vàng đã gán nhãn thủ công
- **THEN** mọi câu được phân đúng lớp mong đợi

#### Scenario: Kết quả tất định
- **WHEN** dự đoán cùng một văn bản năm lần
- **THEN** cả năm lần cho cùng nhãn và cùng độ tin cậy

#### Scenario: Không hồi quy
- **WHEN** so dự đoán hiện tại với file dự đoán vàng đã lưu
- **THEN** kết quả khớp trong phạm vi sai số cho phép

### Requirement: Kiểm chứng OOD phân biệt domain chứ không phải văn phong
Bộ test SHALL bao gồm trường hợp văn bản dài, trang trọng nhưng nằm ngoài miền tin tức.

#### Scenario: Văn bản formal ngoài miền
- **WHEN** đưa vào một đoạn công thức nấu ăn dài và một đoạn văn bản pháp lý
- **THEN** kết quả được ghi lại; nếu detector không bắt được, hạn chế này phải được nêu rõ trong tài liệu

### Requirement: Tiny model fixture cho CI
Bộ test SHALL cung cấp fixture model rất nhỏ dùng cho unit và integration test.

#### Scenario: Fixture nhẹ
- **WHEN** integration test cần một model transformer
- **THEN** fixture cung cấp model nhỏ dưới 10MB thay vì checkpoint đầy đủ

### Requirement: Smoke test toàn hệ thống qua compose
Bộ test SHALL có một kịch bản khởi động toàn bộ stack và kiểm chứng đường đi thực tế của một request.

#### Scenario: Smoke test thành công
- **WHEN** khởi động compose, chờ healthy, gọi `/health`, `/ready`, `/v1/predict`, `/metrics`
- **THEN** mọi lời gọi thành công và metrics phản ánh request vừa gửi

#### Scenario: Thu log khi thất bại
- **WHEN** smoke test thất bại
- **THEN** log của mọi container được thu thập và in ra để chẩn đoán
