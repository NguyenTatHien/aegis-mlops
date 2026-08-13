## ADDED Requirements

### Requirement: Trang demo phục vụ từ chính container API
Hệ thống SHALL phục vụ một trang demo tĩnh tại đường dẫn gốc `/` từ chính service `api`. MUST NOT thêm service mới vào compose và MUST NOT yêu cầu bước build phía client.

#### Scenario: Mở trang gốc
- **WHEN** truy cập `http://localhost:8000/`
- **THEN** trang demo hiển thị, không có lời gọi tài nguyên nào ra ngoài internet

#### Scenario: Không ảnh hưởng route sẵn có
- **WHEN** truy cập `/docs`, `/metrics`, `/health` và `/v1/predict` sau khi mount trang tĩnh
- **THEN** mọi route giữ nguyên hành vi như trước

#### Scenario: Không thêm dependency runtime
- **WHEN** kiểm tra `requirements.txt` và image cuối
- **THEN** không có Node, bundler, hay thư viện JavaScript tải từ CDN nào được thêm vào

### Requirement: Trang demo là client thuần của API công khai
Trang SHALL lấy toàn bộ dữ liệu hiển thị từ `/v1/predict` và `/v1/model/info`. MUST NOT chứa logic phân loại, ngưỡng OOD, hay danh sách tên nhãn hardcode trong JavaScript.

#### Scenario: Tên nhãn lấy từ API
- **WHEN** trang khởi tạo
- **THEN** danh sách nhãn hiển thị được lấy từ `/v1/model/info`, không nhúng cứng trong mã nguồn trang

#### Scenario: Ngưỡng OOD lấy từ API
- **WHEN** trang hiển thị trạng thái OOD
- **THEN** ngưỡng và tên phương pháp lấy từ trường `ood` của response, không tính lại phía client

### Requirement: Nhập văn bản và chọn nhánh model
Trang SHALL cho phép nhập văn bản tự do và chọn giữa hai nhánh `baseline` và `roberta`, mặc định `roberta`.

#### Scenario: Phân loại bằng nhánh mặc định
- **WHEN** nhập một bài tin thể thao và bấm nút phân loại mà không đổi lựa chọn model
- **THEN** trang gọi `/v1/predict` với `model=roberta` và hiển thị kết quả

#### Scenario: Đổi sang nhánh baseline
- **WHEN** chọn `baseline` rồi phân loại cùng văn bản đó
- **THEN** trang gọi `/v1/predict?model=baseline` và hiển thị kết quả của nhánh baseline

### Requirement: Hiển thị kết quả phân loại
Trang SHALL hiển thị nhãn dự đoán, độ tin cậy dạng trực quan, độ trễ đo được và version model đang phục vụ.

#### Scenario: Kết quả đầy đủ
- **WHEN** một lời gọi phân loại thành công
- **THEN** trang hiển thị tên nhãn, thanh biểu diễn độ tin cậy kèm giá trị phần trăm, `latency_ms`, và `model_version`

#### Scenario: Nhãn hiển thị là tên thật
- **WHEN** xem nhãn trên trang
- **THEN** giá trị thuộc {`World`, `Sports`, `Business`, `Sci/Tech`} và không bao giờ khớp mẫu `LABEL_\d`

### Requirement: Hiển thị trạng thái OOD
Trang SHALL hiển thị rõ trạng thái Out-of-Domain, và MUST xử lý được cả trường hợp tính năng OOD đang tắt.

#### Scenario: Văn bản trong miền
- **WHEN** phân loại một bài tin tức hợp lệ và `ood.is_ood` bằng `false`
- **THEN** trang hiển thị trạng thái bình thường kèm điểm bất thường và ngưỡng

#### Scenario: Văn bản ngoài miền
- **WHEN** phân loại một đoạn quảng cáo và `ood.is_ood` bằng `true`
- **THEN** trang hiển thị cảnh báo nổi bật cho biết văn bản cần chuyển sang người kiểm duyệt

#### Scenario: Tính năng OOD đang tắt
- **WHEN** `OOD_ENABLED=false` khiến trường `ood` trong response bằng `null`
- **THEN** trang hiển thị rõ trạng thái OOD đang tắt và vẫn hiển thị đầy đủ kết quả phân loại, không báo lỗi

### Requirement: So sánh hai model tại chỗ
Trang SHALL cho phép so sánh kết quả của hai nhánh model trên cùng một văn bản mà không phải nhập lại.

#### Scenario: So sánh trên cùng đầu vào
- **WHEN** phân loại một văn bản bằng một nhánh rồi chuyển sang nhánh còn lại
- **THEN** văn bản đã nhập được giữ nguyên và trang hiển thị được kết quả cùng độ trễ của cả hai nhánh để đối chiếu

### Requirement: Xử lý lỗi trên giao diện
Trang SHALL hiển thị thông báo dễ hiểu khi API trả lỗi. MUST NOT để trang treo im lặng hay in stack trace.

#### Scenario: Gửi văn bản rỗng
- **WHEN** bấm phân loại khi ô nhập đang trống
- **THEN** trang hiển thị thông báo yêu cầu nhập văn bản và không gửi request

#### Scenario: API chưa sẵn sàng
- **WHEN** API trả HTTP 503 vì model chưa nạp xong
- **THEN** trang hiển thị thông báo hệ thống đang khởi động và gợi ý thử lại

#### Scenario: Lỗi phía server
- **WHEN** API trả HTTP 500
- **THEN** trang hiển thị thông báo lỗi kèm `request_id` để tra log, không in stack trace

### Requirement: Trang demo được kiểm chứng tự động
Bộ test SHALL kiểm chứng trang demo được phục vụ đúng và không phá vỡ các route sẵn có.

#### Scenario: Trang được phục vụ
- **WHEN** integration test gọi `GET /`
- **THEN** trả HTTP 200 với content type HTML

#### Scenario: Không có tham chiếu ra ngoài
- **WHEN** kiểm tra nội dung trang
- **THEN** không có thẻ `script` hay `link` nào trỏ tới host bên ngoài

#### Scenario: Smoke test bao gồm trang demo
- **WHEN** chạy smoke test toàn hệ thống qua compose
- **THEN** `GET /` nằm trong danh sách endpoint được kiểm chứng
