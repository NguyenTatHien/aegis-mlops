## ADDED Requirements

### Requirement: Endpoint metrics theo chuẩn Prometheus
API SHALL phơi `GET /metrics` theo định dạng Prometheus exposition.

#### Scenario: Định dạng hợp lệ
- **WHEN** GET `/metrics`
- **THEN** trả HTTP 200 với content type Prometheus và nội dung parse được

### Requirement: Metrics hệ thống
Hệ thống SHALL thu thập số lượng request, độ trễ và số lỗi, có nhãn phân biệt endpoint, mã trạng thái và model.

#### Scenario: Đếm request
- **WHEN** gọi `/v1/predict` ba lần thành công
- **THEN** `prediction_requests_total` với nhãn tương ứng tăng đúng 3

#### Scenario: Histogram độ trễ có bucket phù hợp SLO
- **WHEN** kiểm tra `prediction_latency_seconds`
- **THEN** danh sách bucket bao gồm mốc 0.5 giây, khớp với cam kết latency dưới 500ms

#### Scenario: Đếm lỗi
- **WHEN** một request gây lỗi phía server
- **THEN** `http_request_errors_total` tăng và nhãn ghi đúng loại lỗi

### Requirement: Metrics đặc thù ML
Hệ thống SHALL thu thập phân bố lớp dự đoán, phân bố độ tin cậy, số lần phát hiện OOD, tỷ lệ OOD và độ dài văn bản đầu vào.

#### Scenario: Phân bố lớp
- **WHEN** thực hiện các lời gọi dự đoán
- **THEN** `predictions_by_class_total` có nhãn `predicted_class` cho từng lớp

#### Scenario: Metrics OOD tồn tại kể cả khi tắt
- **WHEN** `OOD_ENABLED=false` và truy vấn `/metrics`
- **THEN** `ood_detected_total` vẫn được đăng ký và giữ giá trị 0

#### Scenario: Thông tin model được phơi ra
- **WHEN** truy vấn `/metrics`
- **THEN** có metric mô tả model đang phục vụ kèm nhãn version và macro-F1

### Requirement: Prometheus thu thập từ API
Prometheus SHALL được cấu hình scrape endpoint metrics của API và nạp file alert rules.

#### Scenario: Target lành mạnh
- **WHEN** hệ thống đã chạy và mở trang targets của Prometheus
- **THEN** target `aegis-api` ở trạng thái `UP`

#### Scenario: Rules được nạp
- **WHEN** kiểm tra trang rules của Prometheus
- **THEN** mọi alert rule đã định nghĩa đều xuất hiện và không có lỗi cú pháp

### Requirement: Alert rules gắn với success metrics đã cam kết
Hệ thống SHALL định nghĩa alert cho độ trễ vượt ngưỡng, tỷ lệ lỗi vượt ngưỡng, API chết, tỷ lệ OOD tăng đột biến và độ tin cậy trung vị tụt.

#### Scenario: Ngưỡng độ trễ khớp cam kết
- **WHEN** đọc rule độ trễ
- **THEN** rule so p95 của `prediction_latency_seconds` với 0.5 giây

#### Scenario: Ngưỡng tỷ lệ lỗi khớp cam kết
- **WHEN** đọc rule tỷ lệ lỗi
- **THEN** rule so tỷ lệ lỗi với 1%

#### Scenario: Ngưỡng OOD tương thích FPR đã hiệu chỉnh
- **WHEN** đọc rule tỷ lệ OOD
- **THEN** ngưỡng cao hơn FPR đo được sau recalibration, để alert không kêu liên tục ở lưu lượng bình thường

### Requirement: Grafana provisioned as code
Datasource và dashboard của Grafana SHALL được khai báo bằng file cấu hình commit trong repo và mount vào container. MUST NOT phụ thuộc vào thao tác cấu hình thủ công.

#### Scenario: Dashboard có sẵn sau khi khởi động
- **WHEN** chạy `docker compose up` trên máy sạch và đăng nhập Grafana
- **THEN** dashboard Aegis đã tồn tại, không cần import thủ công

#### Scenario: Datasource đã kết nối
- **WHEN** kiểm tra datasource trong Grafana
- **THEN** datasource Prometheus tồn tại và kiểm tra kết nối thành công

### Requirement: Dashboard hiển thị đủ ba nhóm chỉ số
Dashboard SHALL có nhóm chỉ số hệ thống, nhóm chỉ số ML, và nhóm thông tin model.

#### Scenario: Nhóm chỉ số hệ thống
- **WHEN** mở dashboard
- **THEN** có panel cho throughput, độ trễ theo phân vị, tỷ lệ lỗi và uptime

#### Scenario: Nhóm chỉ số ML
- **WHEN** mở dashboard
- **THEN** có panel phân bố lớp dự đoán, phân bố độ tin cậy, và tỷ lệ OOD theo thời gian kèm đường ngưỡng

#### Scenario: So sánh hai model
- **WHEN** mở dashboard
- **THEN** panel độ trễ tách theo nhãn `model`, hiển thị được đồng thời baseline và roberta

#### Scenario: Trạng thái OOD hiển thị rõ
- **WHEN** `OOD_ENABLED=false`
- **THEN** dashboard hiển thị rõ trạng thái OOD đang tắt thay vì panel trống không giải thích
