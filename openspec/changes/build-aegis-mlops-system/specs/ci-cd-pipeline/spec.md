## ADDED Requirements

### Requirement: CI chạy tự động trên mọi pull request
Repo SHALL có workflow GitHub Actions kích hoạt ở mọi pull request và mọi push vào nhánh chính.

#### Scenario: Mở pull request
- **WHEN** một pull request được mở
- **THEN** workflow CI khởi chạy và trạng thái hiển thị trên pull request

### Requirement: Kiểm tra chất lượng mã
CI SHALL chạy kiểm tra lint, định dạng, kiểu tĩnh và quét bảo mật mã nguồn.

#### Scenario: Vi phạm lint
- **WHEN** mã có lỗi lint hoặc sai định dạng
- **THEN** job kiểm tra chất lượng fail và pipeline dừng trước bước test

#### Scenario: Quét bảo mật
- **WHEN** job chất lượng chạy
- **THEN** công cụ quét bảo mật tĩnh chạy trên thư mục source và báo cáo kết quả

### Requirement: Job test có độ phủ
CI SHALL chạy unit test và integration test kèm đo độ phủ, và fail khi dưới ngưỡng.

#### Scenario: Test fail
- **WHEN** bất kỳ test nào fail
- **THEN** job test fail và các job phụ thuộc không chạy

#### Scenario: Báo cáo độ phủ
- **WHEN** job test hoàn tất
- **THEN** báo cáo độ phủ được tải lên như một artifact của workflow

### Requirement: CI không tải model đầy đủ
Job test trên pull request SHALL chạy bằng tiny model fixture. MUST NOT tải checkpoint RoBERTa đầy đủ.

#### Scenario: Không tải checkpoint lớn
- **WHEN** job test chạy trên runner
- **THEN** không có lượt tải nào vượt quá 100MB trong bước test

#### Scenario: Cache dependency
- **WHEN** job test chạy lần thứ hai với dependency không đổi
- **THEN** bước cài đặt dùng lại cache thay vì tải lại từ đầu

### Requirement: Job build và quét image
CI SHALL build Docker image và quét lỗ hổng sau khi test xanh.

#### Scenario: Build sau khi test xanh
- **WHEN** job test thành công
- **THEN** job build image chạy và hoàn tất

#### Scenario: Phát hiện lỗ hổng nghiêm trọng
- **WHEN** quét image phát hiện lỗ hổng mức cao hoặc nghiêm trọng
- **THEN** job build fail

### Requirement: Smoke test compose trong CI
CI SHALL khởi động toàn bộ stack bằng compose và kiểm chứng các endpoint chính.

#### Scenario: Smoke test xanh
- **WHEN** job smoke chạy
- **THEN** stack khởi động, `/health` và `/v1/predict` trả về thành công, rồi stack được dọn sạch

#### Scenario: Thu log khi fail
- **WHEN** job smoke fail
- **THEN** log của mọi container được in vào output của workflow

### Requirement: Model validation tách workflow riêng
Model validation test SHALL chạy ở workflow riêng theo lịch và kích hoạt thủ công được. MUST NOT chạy ở mọi pull request.

#### Scenario: Chạy theo lịch
- **WHEN** đến giờ đã hẹn
- **THEN** workflow model validation chạy với model đầy đủ và báo cáo kết quả

#### Scenario: Kích hoạt thủ công
- **WHEN** thành viên nhóm bấm chạy thủ công
- **THEN** workflow khởi chạy trên nhánh được chọn

### Requirement: Bảo vệ nhánh chính
Nhánh chính SHALL chỉ nhận thay đổi qua pull request đã được duyệt và có CI xanh.

#### Scenario: Chặn push thẳng
- **WHEN** ai đó push thẳng vào nhánh chính
- **THEN** thao tác bị từ chối

#### Scenario: Chặn merge khi CI đỏ
- **WHEN** pull request có CI fail
- **THEN** nút merge bị chặn

### Requirement: Pre-commit hook chặn file lớn
Repo SHALL có cấu hình pre-commit chặn commit file vượt quá giới hạn kích thước.

#### Scenario: Commit nhầm checkpoint
- **WHEN** ai đó thử commit file model 498MB
- **THEN** pre-commit hook chặn lại và báo lỗi rõ ràng

### Requirement: Loại trừ artifact nặng khỏi Git
`.gitignore` SHALL loại trừ `roberta_checkpoints/`, `*.zip`, file model đã train và thư mục cache dữ liệu.

#### Scenario: Trạng thái repo sạch
- **WHEN** chạy pipeline sinh artifact rồi kiểm tra trạng thái Git
- **THEN** không có file artifact nặng nào xuất hiện như thay đổi chưa theo dõi

### Requirement: Đóng góp của mọi thành viên hiển thị trong lịch sử Git
Repo SHALL có commit ý nghĩa từ cả năm thành viên, kèm tài liệu phân công vai trò.

#### Scenario: Kiểm tra phân bố commit
- **WHEN** chạy thống kê tác giả commit
- **THEN** cả năm thành viên đều có commit, và `CONTRIBUTING.md` mô tả trách nhiệm của từng người
