## ADDED Requirements

### Requirement: Endpoint dự đoán có versioning
API SHALL cung cấp `POST /v1/predict` nhận một văn bản và trả về nhãn dự đoán kèm độ tin cậy. Mọi endpoint nghiệp vụ MUST nằm dưới tiền tố `/v1`.

#### Scenario: Dự đoán thành công
- **WHEN** POST `/v1/predict` với body `{"text": "The national team won the championship last night."}`
- **THEN** trả HTTP 200 với `predicted_class`, `confidence`, `model`, `ood`, `latency_ms`, `model_version`

#### Scenario: Nhãn là tên thật, không phải LABEL_x
- **WHEN** nhận bất kỳ response dự đoán nào
- **THEN** `predicted_class` thuộc {`World`, `Sports`, `Business`, `Sci/Tech`} và không bao giờ khớp mẫu `LABEL_\d`

### Requirement: Chọn model qua tham số truy vấn
API SHALL hỗ trợ `?model=baseline|roberta`, mặc định là `roberta`.

#### Scenario: Dùng nhánh baseline
- **WHEN** POST `/v1/predict?model=baseline`
- **THEN** response có `model` bằng `baseline` và độ trễ thấp hơn đáng kể so với nhánh roberta

#### Scenario: Mặc định khi không chỉ định
- **WHEN** POST `/v1/predict` không kèm tham số `model`
- **THEN** response có `model` bằng `roberta`

#### Scenario: Tên model không hợp lệ
- **WHEN** POST `/v1/predict?model=gpt4`
- **THEN** trả HTTP 422 kèm thông báo liệt kê các giá trị hợp lệ

### Requirement: Dự đoán theo lô
API SHALL cung cấp `POST /v1/predict/batch` nhận danh sách văn bản, giới hạn số lượng tối đa cấu hình được.

#### Scenario: Lô hợp lệ
- **WHEN** POST danh sách 10 văn bản
- **THEN** trả về 10 kết quả theo đúng thứ tự đầu vào

#### Scenario: Vượt giới hạn lô
- **WHEN** POST danh sách vượt quá giới hạn cấu hình
- **THEN** trả HTTP 422 nêu rõ giới hạn

### Requirement: Validate đầu vào
API SHALL từ chối văn bản rỗng, chỉ chứa khoảng trắng, hoặc vượt quá độ dài tối đa.

#### Scenario: Text rỗng
- **WHEN** POST `/v1/predict` với `{"text": ""}`
- **THEN** trả HTTP 422 kèm thông báo lỗi mô tả được vấn đề

#### Scenario: Chỉ có khoảng trắng
- **WHEN** POST với `{"text": "   \n\t  "}`
- **THEN** trả HTTP 422

#### Scenario: Thiếu trường bắt buộc
- **WHEN** POST với body `{}`
- **THEN** trả HTTP 422 chỉ rõ trường `text` bị thiếu

### Requirement: Health và readiness tách biệt
API SHALL cung cấp `GET /health` (tiến trình còn sống) và `GET /ready` (model đã nạp xong).

#### Scenario: Sống nhưng chưa sẵn sàng
- **WHEN** tiến trình đã chạy nhưng model chưa nạp xong
- **THEN** `/health` trả HTTP 200 và `/ready` trả HTTP 503

#### Scenario: Sẵn sàng phục vụ
- **WHEN** model đã nạp xong
- **THEN** cả `/health` và `/ready` đều trả HTTP 200

### Requirement: Endpoint thông tin model
API SHALL cung cấp `GET /v1/model/info` mô tả model đang phục vụ.

#### Scenario: Truy vấn thông tin
- **WHEN** GET `/v1/model/info`
- **THEN** response chứa tên model, version, macro-F1, `ood_enabled`, ngưỡng OOD đang dùng, `max_len` và danh sách tên nhãn

### Requirement: Tài liệu OpenAPI tự động
API SHALL phơi OpenAPI spec tại `/openapi.json` và giao diện Swagger tại `/docs`. Mọi endpoint MUST có description và ví dụ.

#### Scenario: Spec hợp lệ
- **WHEN** GET `/openapi.json`
- **THEN** trả về OpenAPI spec hợp lệ liệt kê đủ mọi endpoint

#### Scenario: Mỗi endpoint có mô tả
- **WHEN** kiểm tra từng path trong spec
- **THEN** mỗi path có `summary` khác rỗng và ít nhất một request example

### Requirement: Xử lý lỗi nhất quán
Mọi phản hồi lỗi SHALL có cùng cấu trúc, kèm `request_id` để truy vết.

#### Scenario: Model chưa nạp
- **WHEN** gọi `/v1/predict` trước khi model nạp xong
- **THEN** trả HTTP 503 với body chứa `error`, `detail`, `request_id`

#### Scenario: Lỗi không lường trước
- **WHEN** predictor ném exception ngoài dự kiến
- **THEN** trả HTTP 500 với `request_id`, và log server ghi lại đúng `request_id` đó, không rò rỉ stack trace ra client

### Requirement: Model nạp một lần lúc khởi động
API SHALL nạp model một lần duy nhất ở giai đoạn startup. MUST NOT nạp lại model ở mỗi request.

#### Scenario: Nạp một lần
- **WHEN** gửi 50 request liên tiếp tới `/v1/predict`
- **THEN** log chỉ ghi nhận đúng một sự kiện nạp model

### Requirement: Predictor inject được để test
API SHALL lấy predictor qua cơ chế dependency injection để test có thể thay bằng bản mock.

#### Scenario: Ghi đè bằng mock
- **WHEN** integration test ghi đè dependency predictor bằng `MockPredictor`
- **THEN** toàn bộ endpoint hoạt động bình thường mà không nạp bất kỳ model thật nào
