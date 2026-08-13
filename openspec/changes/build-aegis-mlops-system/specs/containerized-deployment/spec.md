## ADDED Requirements

### Requirement: Dockerfile multi-stage
API SHALL được đóng gói bằng Dockerfile multi-stage tách giai đoạn build và runtime, dùng base image slim.

#### Scenario: Build thành công
- **WHEN** chạy `docker build -f docker/Dockerfile .`
- **THEN** build hoàn tất và image chạy được API

#### Scenario: Runtime không chứa build tool
- **WHEN** kiểm tra image cuối
- **THEN** image không chứa compiler hay công cụ build của giai đoạn builder

### Requirement: Container chạy bằng user không phải root
Container SHALL chạy dưới user không đặc quyền.

#### Scenario: Kiểm tra user
- **WHEN** chạy `docker run --rm <image> id -u`
- **THEN** kết quả khác `0`

### Requirement: Torch bản CPU
Image SHALL cài `torch` bản CPU-only. MUST NOT kéo về gói CUDA runtime.

#### Scenario: Không có CUDA
- **WHEN** kiểm tra danh sách package trong image
- **THEN** không có package `nvidia-*` hay `cuda-*` nào được cài

### Requirement: Loại trừ artifact nặng khỏi build context
`.dockerignore` SHALL loại `roberta_checkpoints/`, `*.zip`, `notebooks/`, `.git/` và các thư mục cache.

#### Scenario: Build context gọn
- **WHEN** chạy docker build
- **THEN** build context không chứa `roberta_checkpoints/` và `aegis_artifacts.zip`

### Requirement: Compose điều phối bốn service
`docker-compose.yml` SHALL định nghĩa các service `api`, `mlflow`, `prometheus`, `grafana` trên một mạng dùng chung, mỗi service có named volume nếu cần lưu trạng thái.

#### Scenario: Khởi động toàn hệ thống
- **WHEN** chạy `docker compose up -d`
- **THEN** cả bốn container đạt trạng thái running

#### Scenario: Cổng được phơi đúng
- **WHEN** hệ thống đã chạy
- **THEN** API trả lời ở cổng 8000, MLflow ở 5001, Prometheus ở 9090, Grafana ở 3000

### Requirement: Health check và thứ tự khởi động
Mỗi service SHALL khai báo health check, và `api` MUST chờ dependency đạt trạng thái healthy trước khi khởi động.

#### Scenario: Chờ dependency
- **WHEN** chạy `docker compose up`
- **THEN** container `api` chỉ khởi động sau khi `mlflow` báo healthy

#### Scenario: Health check phản ánh đúng trạng thái
- **WHEN** chạy `docker compose ps`
- **THEN** cột trạng thái của mỗi service hiển thị `healthy` chứ không chỉ `running`

### Requirement: Chiến lược nạp model cấu hình được
Hệ thống SHALL hỗ trợ `MODEL_SOURCE=registry|local`. Chế độ `registry` nạp từ MLflow, chế độ `local` nạp từ đường dẫn artifact trong image hoặc volume.

#### Scenario: Nạp từ registry
- **WHEN** `MODEL_SOURCE=registry` và MLflow đang chạy
- **THEN** API nạp model từ Model Registry và `/ready` trả HTTP 200

#### Scenario: Lùi về local khi registry lỗi
- **WHEN** `MODEL_SOURCE=local`
- **THEN** API nạp model từ đĩa và phục vụ được mà không cần MLflow

### Requirement: Cấu hình qua biến môi trường
Hệ thống SHALL đọc mọi cấu hình vận hành từ biến môi trường và cung cấp file `.env.example`. MUST NOT hardcode cổng, đường dẫn hay thông tin đăng nhập trong source.

#### Scenario: Có file mẫu
- **WHEN** kiểm tra repo
- **THEN** `.env.example` tồn tại và liệt kê mọi biến bắt buộc kèm giá trị mặc định an toàn

### Requirement: Giới hạn tài nguyên container
Service `api` SHALL khai báo giới hạn bộ nhớ đủ để chứa RoBERTa.

#### Scenario: Khai báo giới hạn
- **WHEN** đọc `docker-compose.yml`
- **THEN** service `api` có giới hạn bộ nhớ ít nhất 4GB

### Requirement: Khởi động sạch trên máy mới
Hệ thống SHALL khởi động được trên máy chưa từng chạy dự án, không cần bước tải model thủ công.

#### Scenario: Người chấm chạy lần đầu
- **WHEN** clone repo, chạy bước chuẩn bị artifact được ghi trong README, rồi `docker compose up`
- **THEN** `/health`, `/ready` và một lời gọi `/v1/predict` đều thành công
