> Thứ tự nhóm task theo quyết định D12 trong `design.md`: contract → pure functions → API + mock → Docker → model thật → MLflow.
> Nhóm 1–5 không cần GPU và không cần model 498MB, nên chạy song song được ngay từ giờ đầu.
> Nhãn owner theo phân công 5 thành viên: **M1** Data/ML · **M2** MLOps/Tracking · **M3** Backend/API · **M4** DevOps/CI · **M5** QA/Monitoring/RAI.

## 1. Nền móng repo và contract

- [x] 1.1 Khởi tạo Git repo, `.gitignore` loại `roberta_checkpoints/`, `*.zip`, `artifacts/`, `data/cache/`, `*.joblib`, `*.safetensors`, `.npy` — **M4**
- [x] 1.2 Tạo cây thư mục `src/aegis/{data,models,ood,api,serving,eval,explain}`, `tests/{unit,integration,data,model}`, `scripts/`, `docker/`, `monitoring/` — **M4**
- [x] 1.3 Viết `requirements.txt` pin cứng mọi version, `transformers` khớp `transformers_version` trong `roberta_final/config.json`; tách `requirements-dev.txt` — **M4**
- [x] 1.4 Cấu hình `pyproject.toml`: ruff, mypy, pytest marker (`unit`/`integration`/`data`/`model`/`slow`), coverage `fail_under = 80` — **M4**
- [x] 1.5 Thêm `.pre-commit-config.yaml` có `check-added-large-files` chặn commit vượt 10MB — **M4**
- [x] 1.6 Chuyển `aegis_ag_news_training (1).ipynb` vào `notebooks/`, đổi tên bỏ khoảng trắng và hậu tố `(1)` — **M1**
- [x] 1.7 Viết `src/aegis/config.py` bằng pydantic-settings: đọc `.env` + `ood_config.json`, phơi `max_len`, `label_names`, ngưỡng, `MODEL_SOURCE`, `OOD_ENABLED`, `MODEL_DEFAULT` — **M3**
- [x] 1.8 Viết `.env.example` liệt kê mọi biến bắt buộc kèm mặc định an toàn — **M4**
- [x] 1.9 Viết `src/aegis/api/schemas.py`: `PredictRequest`, `PredictResponse` (có sẵn `ood: Optional[OODResult] = None`), `OODResult`, `ErrorResponse`, `ModelInfoResponse` — **M3**
- [x] 1.10 Định nghĩa `Predictor` Protocol và `OODDetector` Protocol (chỉ chữ ký, chưa implement) — **M3**
- [x] 1.11 Viết `tests/unit/test_config.py`: khẳng định `max_len == 128`, `label_names` đúng 4 phần tử, ngưỡng đọc từ file không hardcode — **M5**
- [x] 1.12 Viết `tests/unit/test_schemas.py`: text rỗng, chỉ khoảng trắng, thiếu trường, `model` không hợp lệ, `ood` nhận `None` — **M5**

## 2. Pure functions — không cần model

- [x] 2.1 Viết `src/aegis/data/preprocess.py` với `clean_text_tfidf()` (bê từ notebook cell 17) và `passthrough()` — **M1**
- [x] 2.2 Viết `src/aegis/models/labels.py`: ánh xạ id ↔ tên nhãn, nguồn duy nhất là `ood_config.json`, không đọc `config.json` — **M1**
- [x] 2.3 Viết `src/aegis/ood/scoring.py`: `msp_score()`, `energy_score()`, `entropy_score()` — hàm thuần trên `np.ndarray` — **M1**
- [x] 2.4 Viết `src/aegis/ood/detector.py`: `MSPDetector`, `EnergyDetector`, `EntropyDetector`, `NullOODDetector`; đọc ngưỡng và `T` từ config — **M1**
- [x] 2.5 Viết `tests/unit/test_preprocess.py`: khẳng định nhánh RoBERTa nhận text thô, nhánh TF-IDF đã làm sạch — **M5**
- [x] 2.6 Viết `tests/unit/test_label_mapping.py`: 0→World, 1→Sports, 2→Business, 3→Sci/Tech; không giá trị nào khớp `LABEL_\d` — **M5**
- [x] 2.7 Viết `tests/unit/test_ood_scoring.py`: MSP trên logits đều bằng 0.75; Energy khớp `-logsumexp`; entropy đều bằng 1.0, chắc chắn bằng 0.0 — **M5**
- [x] 2.8 Viết `tests/unit/test_ood_interface.py` parametrize qua cả bốn detector, dùng logits giả — **M5**
- [x] 2.9 Viết `tests/unit/test_threshold.py`: logic `is_ood`, biên khi score bằng đúng ngưỡng, hành vi khi thiếu `ood_config.json` — **M5**

## 3. API với MockPredictor

- [x] 3.1 Viết `src/aegis/serving/mock_predictor.py` trả logits cố định, dùng cho test và phát triển song song — **M3**
- [x] 3.2 Viết `src/aegis/api/dependencies.py`: cung cấp predictor qua `Depends()`, ghi đè được trong test — **M3**
- [x] 3.3 Viết `src/aegis/api/main.py`: `lifespan` nạp model một lần, middleware `request_id`, structured JSON logging — **M3**
- [x] 3.4 Viết route `POST /v1/predict` có `?model=baseline|roberta`, mặc định `roberta` — **M3**
- [x] 3.5 Viết route `POST /v1/predict/batch` có giới hạn số phần tử cấu hình được — **M3**
- [x] 3.6 Viết route `GET /health` và `GET /ready` (tách biệt: ready chỉ 200 khi model đã nạp) — **M3**
- [x] 3.7 Viết route `GET /v1/model/info` trả version, macro-F1, `ood_enabled`, ngưỡng, `max_len`, `label_names` — **M3**
- [x] 3.8 Viết exception handler thống nhất: 422, 503, 500 đều có `error`/`detail`/`request_id`, không rò stack trace — **M3**
- [x] 3.9 Bổ sung `summary`, `description` và request example cho mọi endpoint trong OpenAPI — **M3**
- [x] 3.10 Chừa chỗ cho `POST /v1/explain` (schema + route trả 501 tạm thời) để mục Responsible AI cắm vào sau — **M3**
- [x] 3.11 Viết `tests/integration/test_api_endpoints.py` phủ mọi endpoint bằng MockPredictor — **M5**
- [x] 3.12 Viết `tests/integration/test_api_errors.py` phủ mọi đường lỗi — **M5**
- [x] 3.13 Viết `tests/integration/test_openapi.py`: spec hợp lệ, mọi path có `summary` khác rỗng — **M5**

## 4. Metrics và endpoint /metrics

- [x] 4.1 Viết `src/aegis/api/metrics.py` đăng ký metrics hệ thống: `prediction_requests_total`, `prediction_latency_seconds` (bucket có mốc 0.5s), `http_request_errors_total`, `inference_duration_seconds` — **M5**
- [x] 4.2 Đăng ký metrics ML: `predictions_by_class_total`, `prediction_confidence`, `ood_detected_total`, `ood_score`, `ood_rate`, `input_text_length_words`, `model_info` — **M5**
- [x] 4.3 Gắn nhãn `model` vào metrics để tách được baseline và roberta trên cùng dashboard — **M5**
- [x] 4.4 Đăng ký metrics OOD ngay cả khi `OOD_ENABLED=false`, giữ giá trị 0 — **M5**
- [x] 4.5 Thêm route `GET /metrics` theo định dạng Prometheus exposition — **M5**
- [x] 4.6 Viết `tests/integration/test_metrics_endpoint.py`: gọi predict 3 lần rồi khẳng định counter tăng đúng 3 — **M5**

## 5. Docker và Compose

- [x] 5.1 Viết `docker/Dockerfile` multi-stage, base slim, `torch` CPU-only wheel, user không phải root, `HEALTHCHECK` — **M4**
- [x] 5.2 Viết `.dockerignore` loại `roberta_checkpoints/`, `*.zip`, `notebooks/`, `.git/`, cache — **M4**
- [x] 5.3 Viết `docker-compose.yml` với 4 service, mạng dùng chung, named volume, cổng 8000/5001/9090/3000 — **M4**
- [x] 5.4 Thêm health check cho cả 4 service và `depends_on: condition: service_healthy` cho `api` — **M4**
- [x] 5.5 Đặt giới hạn bộ nhớ cho service `api` ít nhất 4GB — **M4**
- [x] 5.6 Implement `MODEL_SOURCE=local` (nạp từ đĩa/volume) và kiểm chứng API chạy được khi MLflow tắt — **M4**
- [x] 5.7 Viết `monitoring/prometheus/prometheus.yml` scrape `api:8000/metrics` mỗi 5s và nạp `alerts.yml` — **M5**
- [x] 5.8 Viết `monitoring/prometheus/alerts.yml`: `HighLatency` (p95 > 0.5s), `HighErrorRate` (> 1%), `APIDown`, `LowConfidence`, `OODSpike` (ngưỡng tạm, chốt lại ở task 7.6) — **M5**
- [x] 5.9 Viết `monitoring/grafana/provisioning/datasources/prometheus.yml` — **M5**
- [x] 5.10 Viết `monitoring/grafana/provisioning/dashboards/aegis.json`: hàng chỉ số hệ thống, hàng chỉ số ML, hàng thông tin model, panel so latency 2 model, panel trạng thái OOD — **M5**
- [x] 5.11 Viết `Makefile`: `make up`, `make down`, `make test`, `make train`, `make lint` — **M4**
- [x] 5.12 Viết `tests/integration/test_compose_smoke.py`: khởi động stack, chờ healthy, gọi `/` `/health` `/ready` `/v1/predict` `/metrics`, thu log container khi fail — **M4**
- [x] 5.13 Viết test khẳng định file provisioning Grafana tồn tại và JSON dashboard parse được — **M5**

## 6. Data pipeline và baseline thật

- [x] 6.1 Viết `src/aegis/data/loader.py`: tải AG News, cache local, chạy được offline ở lần thứ hai — **M1**
- [x] 6.2 Viết `src/aegis/data/split.py`: 90/10 stratify `random_state=42`, ghi index ra file — **M1**
- [x] 6.3 Viết `src/aegis/data/versioning.py`: hash SHA256 + số dòng + phân bố lớp → `data/dataset_card.json`, cảnh báo khi hash lệch — **M1**
- [x] 6.4 Viết `src/aegis/data/validate.py`: schema, miền nhãn, cân bằng lớp, null, khoảng độ dài, trùng lặp, rò rỉ train/test — **M1**
- [x] 6.5 Viết `tests/data/test_data_quality.py` phủ đủ 9 kiểm tra ở spec `automated-testing` — **M5**
- [x] 6.6 Viết `src/aegis/models/train_baseline.py`: TF-IDF + `LogisticRegression`, GridSearchCV trên `C`, lưu `baseline/logreg_model.joblib` — **M1**
- [x] 6.7 Kiểm chứng `predict_proba` trả 4 phần tử tổng bằng 1.0 và ghi lại macro-F1 mới của baseline — **M1**
- [x] 6.8 Viết `src/aegis/eval/evaluate.py`: classification report JSON, confusion matrix PNG, macro-F1 — **M1**
- [x] 6.9 Viết `src/aegis/models/compare.py` sinh lại `model_comparison.json` từ số liệu mới — **M1**

## 7. Model thật và recalibrate OOD

- [x] 7.1 Viết `src/aegis/serving/roberta_predictor.py`: nạp `roberta_final`, text thô, `max_len` từ config, `torch.set_num_threads()`, chạy `eval()` và `no_grad` — **M3**
- [x] 7.2 Viết `src/aegis/serving/baseline_predictor.py`: nạp vectorizer + LogReg, áp `clean_text_tfidf()` — **M3**
- [x] 7.3 Viết test cold-start nạp model offline trong container (không có mạng), cấm dùng slow tokenizer — **M5**
- [x] 7.4 Viết `scripts/collect_logits.py`: chạy inference trên val ID, test ID, OOD val, OOD test (~24k mẫu), cache ra `.npy` — **M1**
- [x] 7.5 Viết `scripts/recalibrate_ood.py`: sweep FPR {1,2,5,10,15,20,30}% × {MSP, Energy} → `ood_operating_points.json` + biểu đồ ROC — **M1**
- [x] 7.6 Nhóm họp chốt operating point trên bảng số thật; ghi quyết định và lý do vào `design.md` mục Open Questions — **cả nhóm**
- [x] 7.7 Sinh `ood_config.json` mới có `target_fpr`, `measured_fpr`, `measured_recall`, `calibrated_at`; giữ file cũ trong lịch sử Git — **M1**
- [x] 7.8 Cập nhật ngưỡng `OODSpike` trong `alerts.yml` cho khớp FPR đo được — **M5**
- [x] 7.9 Bật `OOD_ENABLED=true`, chạy lại toàn bộ integration test — **M3**
- [x] 7.10 Viết `tests/model/test_performance.py`: macro-F1 ≥ 0.90, F1 từng lớp ≥ 0.85 — **M5**
- [x] 7.11 Viết `tests/model/test_ood_quality.py`: FPR ≤ 0.10, recall theo giá trị đo được sau recalibration — **M5**
- [x] 7.12 Viết `tests/model/test_latency.py`: p95 một lời gọi dự đoán trên CPU dưới 500ms — **M5**
- [x] 7.13 Viết `tests/model/test_behavior.py`: bất biến khoảng trắng, định hướng trên tập câu vàng, tất định qua 5 lần chạy — **M5**
- [x] 7.14 Sinh `tests/fixtures/golden_predictions.json` và viết test hồi quy so với file này — **M5**
- [x] 7.15 Viết test đối chứng OOD: văn bản dài trang trọng ngoài miền (công thức nấu ăn, đoạn pháp lý); ghi kết quả vào tài liệu dù đạt hay không — **M5**

## 8. MLflow

- [x] 8.1 Thêm service MLflow vào compose: cổng 5001, named volume cho backend store và artifact store — **M2**
- [x] 8.2 Tích hợp MLflow vào `train_baseline.py`: log params, metrics, artifacts; mỗi giá trị `C` là một run — **M2**
- [x] 8.3 Chạy training baseline live và kiểm chứng ít nhất 7 run xuất hiện trong MLflow UI — **M2**
- [x] 8.4 Viết `scripts/backfill_mlflow.py` đọc `roberta_checkpoints/*/trainer_state.json` → tạo run RoBERTa, gắn tag `source=backfill` và đường dẫn checkpoint gốc — **M2**
- [x] 8.5 Log kết quả recalibration OOD lên MLflow kèm artifact `ood_operating_points.json` và biểu đồ ROC — **M2**
- [x] 8.6 Đăng ký `aegis-baseline` và `aegis-roberta` vào Model Registry, gán stage `Production` — **M2**
- [x] 8.7 Implement `MODEL_SOURCE=registry` trong predictor, ghi log version đã nạp — **M2**
- [x] 8.8 Kiểm chứng dữ liệu MLflow còn nguyên sau khi restart container — **M2**
- [x] 8.9 Viết `tests/integration/test_mlflow_integration.py`: nạp model từ registry URI thành công — **M5**

## 9. CI/CD

- [x] 9.1 Viết `.github/workflows/ci.yml` kích hoạt ở pull request và push nhánh chính — **M4**
- [x] 9.2 Thêm job `lint`: ruff check, ruff format kiểm tra, mypy, quét bảo mật tĩnh trên `src/` — **M4**
- [x] 9.3 Thêm job `test`: cache pip và HF hub, chạy `pytest -m "unit or integration"`, coverage gate 80%, tải lên báo cáo coverage — **M4**
- [x] 9.4 Thêm tiny model fixture vào `conftest.py` và kiểm chứng job test không tải file nào vượt 100MB — **M4**
- [x] 9.5 Thêm job `data-tests` chạy `pytest -m data` có cache dataset — **M4**
- [x] 9.6 Thêm job `docker-build`: buildx có layer cache, quét lỗ hổng image, fail khi có lỗ hổng mức cao trở lên — **M4**
- [x] 9.7 Thêm job `compose-smoke` chạy sau `docker-build`, thu log container khi fail — **M4**
- [x] 9.8 Viết `.github/workflows/model-validation.yml`: chạy theo lịch và kích hoạt thủ công, chạy `pytest -m model` với model đầy đủ — **M4**
- [ ] 9.9 Bật branch protection cho nhánh chính: bắt buộc pull request, bắt buộc CI xanh, chặn push thẳng — **M4**
- [x] 9.10 Thêm PR template và `CODEOWNERS` — **M4**
- [ ] 9.11 Kiểm chứng lịch sử Git có commit ý nghĩa từ cả 5 thành viên — **cả nhóm**

## 10. Trang demo tĩnh

> Làm được ngay sau nhóm 3 vì trang chỉ cần API chạy với MockPredictor. Không đụng vào `docker-compose.yml`.

- [x] 10.1 Tạo `src/aegis/api/static/index.html`: một file, không CDN, không bundler — **M3**
- [x] 10.2 Mount `StaticFiles` tại `/` trong `main.py`, kiểm chứng `/docs`, `/metrics`, `/v1/*` giữ nguyên hành vi — **M3**
- [x] 10.3 Dựng form nhập văn bản và bộ chọn nhánh model, mặc định `roberta` — **M3**
- [x] 10.4 Gọi `/v1/model/info` lúc khởi tạo để lấy tên nhãn, `ood_enabled` và version; cấm hardcode nhãn trong JS — **M3**
- [x] 10.5 Hiển thị kết quả: tên nhãn, thanh độ tin cậy kèm phần trăm, `latency_ms`, `model_version` — **M3**
- [x] 10.6 Hiển thị trạng thái OOD gồm ba nhánh: trong miền, ngoài miền có cảnh báo nổi bật, và tính năng đang tắt khi `ood` bằng `null` — **M3**
- [x] 10.7 Giữ nguyên văn bản đã nhập khi đổi model để so sánh hai nhánh tại chỗ — **M3**
- [x] 10.8 Xử lý lỗi trên giao diện: ô nhập trống chặn tại client, 503 báo đang khởi động, 500 hiển thị `request_id` — **M3**
- [x] 10.9 Chuẩn bị sẵn vài văn bản mẫu bấm là điền, gồm một bài tin thật và một đoạn quảng cáo, để demo không phải gõ tay — **M5**
- [x] 10.10 Viết `tests/integration/test_static_page.py`: `GET /` trả HTML, không có thẻ `script` hay `link` trỏ ra host ngoài — **M5**
- [x] 10.11 Kiểm chứng `.dockerignore` không loại nhầm thư mục `static/` và trang phục vụ được từ trong container — **M4**

## 11. Tài liệu và bàn giao

- [x] 11.1 Viết `README.md`: badge CI và coverage, hướng dẫn cài đặt, chuẩn bị artifact, `docker compose up`, ví dụ gọi API, mục xử lý sự cố — **M5**
- [x] 11.2 Viết `ARCHITECTURE.md`: sơ đồ kiến trúc, luồng dữ liệu, luận giải lựa chọn công nghệ, bảng trade-off — **M3**
- [x] 11.3 Viết `CONTRIBUTING.md`: vai trò và trách nhiệm của 5 thành viên, quy ước nhánh, quy ước commit — **M5**
- [ ] 11.4 Sửa `MLOps.docx` mục Scope theo quyết định D2, thay câu về Transformer bằng trade-off analysis có số liệu — **cả nhóm duyệt**
- [ ] 11.5 Bổ sung trần FPR và ngưỡng OOD đã chốt vào mục Success Metrics của `MLOps.docx` — **M1**
- [x] 11.6 Đưa `ood_operating_points.json` và biểu đồ ROC vào tài liệu làm bằng chứng cho phần Responsible AI — **M5**
- [ ] 11.7 Chạy thử toàn bộ trên máy sạch: clone, chuẩn bị artifact theo README, `docker compose up`, mở trang demo, gọi predict, mở Grafana — **cả nhóm**
- [x] 11.8 Ghi lại hạn chế đã biết vào README: OOD dùng dataset proxy, trade-off FPR và recall, kết quả test đối chứng ở task 7.15 — **M5**
- [ ] 11.9 Tổng duyệt Live Demo theo kịch bản: trang demo phân loại đúng → dán spam thấy cảnh báo OOD → đổi model so latency → chuyển sang Grafana thấy panel phản ứng — **cả nhóm**
