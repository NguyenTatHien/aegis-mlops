## Why

Model đã train xong (RoBERTa macro-F1 0.9517, TF-IDF baseline 0.9259, OOD MSP/Energy đã calibrate), nhưng toàn bộ hệ thống bao quanh nó vẫn chưa tồn tại: chưa có API, chưa có container, chưa có MLflow, chưa có monitoring, chưa có một dòng test nào. Theo rubric DDM501, phần model chỉ chiếm 15/40 điểm của mục C — nghĩa là ~51/55 điểm của mục C + D (Implementation + Testing & CI/CD) hiện đang bỏ trống.

Đồng thời có ba khiếm khuyết đã được xác minh cần sửa cùng lúc, vì để càng lâu càng đắt: ngưỡng OOD hiện tại loại nhầm 35–41% tin tức hợp lệ; `aegis_predict()` áp preprocessing của TF-IDF lên RoBERTa (train/serve skew); và baseline `LinearSVC` không có `predict_proba` nên không thể trả `confidence` như User Requirement #3 yêu cầu.

## What Changes

- **Chuyển notebook thành source tree có thể test được**: tách `src/aegis/` thành data / models / ood / api / serving, mỗi module có entrypoint CLI và test riêng.
- **Serving hai model qua một API** (`/v1/predict?model=baseline|roberta`, mặc định `roberta`): giữ trọn giá trị của cả TF-IDF baseline lẫn RoBERTa, biến `model_comparison.json` từ artifact tĩnh thành tính năng chạy được và so sánh trực tiếp trên Grafana.
- **BREAKING (tài liệu)**: `MLOps.docx` mục Scope hiện ghi *"Transformer được triển khai ở mức benchmark... thay vì trở thành kiến trúc chính"*. Câu này phải sửa: RoBERTa là serving model mặc định, vì (a) hơn baseline 2.6 điểm macro-F1 và (b) OOD detection bắt buộc cần logits, mà `LinearSVC` không cung cấp.
- **BREAKING (model)**: thay `LinearSVC` bằng `LogisticRegression` cho nhánh baseline — khôi phục `predict_proba`, mở đường cho entropy-based OOD trên baseline, và khớp lại đúng nguyên văn `MLOps.docx`.
- **Recalibrate ngưỡng OOD** theo operating-point table (sweep FPR ∈ {1, 2, 5, 10, 15, 20, 30}% cho cả MSP và Energy), mặc định chọn FPR ≤ 5%. Ngưỡng cũ (FPR 35–41%) khiến alert Grafana đỏ vĩnh viễn ngay request đầu tiên.
- **Sửa train/serve skew**: nhánh RoBERTa nhận text thô (đúng như lúc train), `clean_text()` chỉ áp cho nhánh TF-IDF.
- **Bổ sung MLflow từ con số 0**: tracking server + Model Registry; backfill các run RoBERTa từ `roberta_checkpoints/*/trainer_state.json` (dữ liệu thật, có log từng epoch) và train lại baseline live để chứng minh pipeline hoạt động.
- **Dựng monitoring stack**: Prometheus (metrics hệ thống + metrics ML tùy chỉnh) → alert rules gắn với Success Metrics đã cam kết (latency < 500ms, error rate < 1%) → Grafana dashboard provisioned as code.
- **Bốn loại test + CI/CD**: unit, integration, data quality, model validation; coverage ≥ 80%; GitHub Actions chạy lint → test → docker build → trivy scan → compose smoke test.
- **Trang demo tĩnh phục vụ ngay trong container `api`**: một file HTML với vanilla JS gọi `fetch`, mount qua `StaticFiles` tại `/`. Không thêm service, không thêm ngôn ngữ vào CI, không có build step. Mục tiêu duy nhất là làm Live Demo kể được trọn câu chuyện phân loại → phát hiện OOD → so hai model, thay vì thao tác qua Swagger UI.

## Capabilities

### New Capabilities

- `ml-pipeline`: Data ingestion, validation, versioning, preprocessing tách đôi theo model, split tái lập được, training baseline + transformer, evaluation và model comparison.
- `experiment-tracking`: MLflow tracking server, logging params/metrics/artifacts, Model Registry với stage, backfill từ checkpoint có sẵn.
- `ood-detection`: Interface `OODDetector` thuần logits, ba implementation (MSP, Energy, Entropy), script recalibration sinh operating-point table và `ood_config.json`.
- `prediction-api`: REST API FastAPI có versioning, hai nhánh model, `/predict`, `/predict/batch`, `/explain`, `/health`, `/ready`, `/model/info`, OpenAPI docs, error handling và dependency injection cho predictor.
- `containerized-deployment`: Dockerfile multi-stage non-root, docker-compose bốn service với health check và điều kiện khởi động, chiến lược nạp model artifact.
- `observability`: Prometheus metrics (system + ML custom), alert rules gắn với success metrics, Grafana datasource và dashboard provisioned as code.
- `automated-testing`: Bốn loại test theo rubric, tiny-model fixture để CI không phải tải 498MB, golden prediction regression, coverage gate 80%.
- `ci-cd-pipeline`: GitHub Actions cho lint/type/security, test + coverage, docker build + scan, compose smoke test, workflow model-validation tách riêng chạy theo lịch.
- `demo-web-ui`: Trang demo một file phục vụ tĩnh từ chính container `api`, cho phép nhập văn bản, chọn nhánh model, xem nhãn kèm độ tin cậy, độ trễ và trạng thái OOD.

### Modified Capabilities

*(Chưa có spec nào trong `openspec/specs/` — đây là change đầu tiên của repo.)*

## Impact

**Code**: Toàn bộ `src/aegis/` là mới. `aegis_ag_news_training.ipynb` được giữ nguyên làm bằng chứng nghiên cứu, chuyển vào `notebooks/`, không còn là nguồn thực thi.

**Artifacts**:
- Dùng lại: `roberta_final/` (498MB), `baseline/tfidf_vectorizer.joblib`, `ood_config.json`, các file JSON kết quả.
- Sinh mới: `baseline/logreg_model.joblib`, `ood_operating_points.json`, `ood_config.json` (phiên bản recalibrated), logits `.npy` cache.
- Loại khỏi Git: `roberta_checkpoints/` (~3.0GB, chứa `optimizer.pt` 997MB × 2) và `aegis_artifacts.zip` (3.1GB) — vượt xa giới hạn 100MB của GitHub.

**Tài liệu**: `MLOps.docx` cần sửa mục Scope (xem BREAKING ở trên) và bổ sung trần FPR vào Success Metrics — hiện chỉ ghi "OOD Detection" chung chung, không có ngưỡng, nên không kiểm chứng được.

**Dependencies mới**: `fastapi`, `uvicorn`, `pydantic-settings`, `mlflow`, `prometheus-client`, `lime`, `pytest`, `pytest-cov`, `ruff`, `mypy`. Trang demo không thêm dependency nào — chỉ là một file tĩnh dùng `fetch` sẵn có của trình duyệt, không Node, không bundler, không CDN. Bắt buộc pin cứng — `roberta_final/config.json` ghi `transformers_version: 5.15.0`, sai version sẽ không load được safetensors.

**Ràng buộc hạ tầng**: GitHub Actions runner free (7GB RAM / 14GB disk) không kham nổi việc tải RoBERTa mỗi PR. Test unit/integration bắt buộc dùng tiny-model fixture; model validation tách sang workflow riêng chạy nightly + `workflow_dispatch`.

**Rủi ro đã biết**: siết FPR xuống 5% kéo OOD recall từ ~89% xuống ~45% (ước lượng từ AUROC 0.859). Đây là trade-off nghiệp vụ do cả nhóm chốt, không phải hyperparameter — operating-point table phải được commit vào repo làm bằng chứng cho phần Responsible AI và Q&A.
