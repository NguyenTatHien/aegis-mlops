## Context

Aegis là bài Final Project của DDM501 (Topic 9 — Document Classification System), làm trên AG News 4 lớp với yêu cầu bổ sung phát hiện Out-of-Domain.

**Trạng thái hiện tại — đã có:**

| Artifact | Kích thước | Kết quả |
|---|---|---|
| `roberta_final/` | 498 MB | test macro-F1 **0.9517** |
| `baseline/tfidf_vectorizer.joblib` + `svm_model.joblib` | 3.6 MB | test macro-F1 **0.9259**, best C = 0.3 |
| `ood_config.json` | — | MSP thr 0.001656, Energy thr −5.6738, T = 1.0, max_len = 128 |
| `roberta_checkpoints/` | ~3.0 GB | `trainer_state.json` có log đầy đủ từng epoch |
| `aegis_ag_news_training.ipynb` | — | 70 cell, pipeline hoàn chỉnh từ EDA đến OOD |

**Trạng thái hiện tại — chưa có gì:** API, Dockerfile, docker-compose, MLflow, Prometheus, Grafana, thư mục `tests/`, thư mục `.github/`.

**Ràng buộc:**
- Thời gian còn lại tính bằng ngày, nhóm 5 người làm song song.
- GitHub Actions runner free: 7 GB RAM, 14 GB disk — không kham nổi việc tải model 498 MB ở mỗi PR.
- GitHub giới hạn file 100 MB — `roberta_final/model.safetensors` (498 MB) không thể commit thẳng.
- Không GPU ở môi trường production; inference chạy CPU.
- `transformers_version: 5.15.0` trong `config.json` — version rất mới, phải pin chính xác.

**Ba khiếm khuyết đã xác minh** (chi tiết ở Decisions D4, D6, D3):
1. Ngưỡng OOD hiện tại: MSP FPR 41.3%, Energy FPR 35.0% — loại nhầm ~1/3 tin tức hợp lệ.
2. `aegis_predict()` (cell 62) gọi `clean_text()` trước khi tokenize cho RoBERTa, trong khi cell 31 train trên text thô.
3. `LinearSVC` không có `predict_proba` → baseline không trả được `confidence` (User Requirement #3).

## Goals / Non-Goals

**Goals:**
- Dựng đủ mục C (Implementation, 40%) và mục D (Testing & CI/CD, 15%) của rubric, ở mức Excellent.
- Tái sử dụng tối đa artifacts đã train — không train lại RoBERTa.
- Cho phép 5 thành viên làm song song từ giờ đầu, không ai bị chặn bởi model 498 MB.
- Mọi trade-off đều có số liệu kèm theo, đủ để trả lời Q&A.
- `docker compose up` chạy được trên máy sạch của người chấm, không cần bước tải model thủ công.

**Non-Goals:**
- Không train lại RoBERTa từ đầu (2–3h GPU, rủi ro lệch số liệu đã viết trong báo cáo).
- Không Kubernetes, không distributed training, không GPU serving.
- Không continuous learning / auto-retraining pipeline.
- Không multi-label, không hierarchical classification.
- Không tối ưu thêm model performance — 0.9517 là đủ; điểm nằm ở phần hệ thống.

## Decisions

### D1 — Serving hai model qua một API, thay vì chọn một

`POST /v1/predict?model=baseline|roberta`, mặc định `roberta`.

*Alternatives:*
- **Chỉ RoBERTa**: đơn giản nhất, khớp artifacts. Nhưng vứt bỏ baseline đã train và làm `model_comparison.json` thành artifact chết.
- **Chỉ baseline TF-IDF**: đúng nguyên văn `MLOps.docx`, image ~250 MB, CI nhanh. Nhưng phải train lại, viết lại OOD bằng entropy, và mất 2.6 điểm F1.

*Lý do chọn:* chi phí thêm khoảng 3 giờ công, đổi lại (a) không vứt gì, (b) `model_comparison.json` trở thành tính năng sống — Grafana so latency 5ms vs 60ms real-time, (c) chính là "trade-off analysis" mà rubric mục B đòi hỏi, ở dạng chạy được chứ không phải bảng trong slide.

### D2 — RoBERTa là serving model mặc định; sửa mục Scope của `MLOps.docx`

Câu *"Transformer được triển khai ở mức benchmark... thay vì trở thành kiến trúc chính"* không còn đúng. Lý do kỹ thuật, không phải đổi ý:
1. RoBERTa hơn baseline 2.6 điểm macro-F1 (0.9517 vs 0.9259).
2. OOD detection (MSP/Energy) cần logits. `LinearSVC` chỉ có `decision_function` chưa hiệu chuẩn ⇒ **User Requirement #4 không khả thi trên baseline hiện tại**.

Đây là quyết định tài liệu chung, cả nhóm phải duyệt trước khi implement.

### D3 — `LogisticRegression` thay `LinearSVC` cho nhánh baseline

*Alternative:* `CalibratedClassifierCV(cv="prefit")` bọc SVM hiện có, hoặc softmax trên `decision_function`.

*Lý do chọn:* train lại LogReg chỉ mất ~1 phút CPU và giải quyết ba việc cùng lúc — có `predict_proba` native, khớp lại đúng nguyên văn `MLOps.docx` (vốn luôn ghi "Logistic Regression"), và mở đường cho entropy-based OOD trên baseline. Softmax trên `decision_function` cho ra số không có ý nghĩa xác suất, không dùng làm `confidence` trả cho user được.

Grid `C` cũ vẫn được log vào MLflow như các experiment bổ sung (rubric đòi *"multiple experiments"*).

### D4 — OOD là post-processing thuần trên logits

```
OODDetector (Protocol)
├── score(logits: np.ndarray) -> float
├── is_ood(score: float) -> bool
├── method: str          # "msp" | "energy" | "entropy"
└── enabled: bool

Implementations: MSPDetector, EnergyDetector (RoBERTa) | EntropyDetector (baseline)
                 NullOODDetector (feature flag off)
```

Detector **không** tokenize, **không** load model, **không** đọc file. Chỉ nhận `logits` vào, trả `score` ra.

*Hệ quả — đây là lý do chính của thiết kế này:*
- Test được bằng `numpy` array giả, không cần tải model 498 MB trong CI.
- API, Dockerfile, compose, Prometheus, dashboard, CI đều không phụ thuộc OOD ⇒ 85–90% mục C và D dựng được song song, không chờ nhau.
- Response schema có sẵn `ood: Optional[OODResult] = None`, nên bật/tắt OOD không phải breaking change.

### D5 — Recalibrate OOD theo operating-point table, không hardcode một ngưỡng

`scripts/recalibrate_ood.py` sweep FPR ∈ {1, 2, 5, 10, 15, 20, 30}% × {MSP, Energy}, xuất `ood_operating_points.json` + `ood_config.json`. Mặc định chọn FPR ≤ 5%.

*Lý do:* ngưỡng cũ tune theo *"Recall ≥ 90%, FPR nhỏ nhất có thể"* — và "nhỏ nhất có thể" hoá ra là 41%. Với FPR đó, alert `OODSpike` sẽ đỏ vĩnh viễn ngay request đầu tiên.

Ngưỡng mới nghiêng về FPR thấp vì OOD **chỉ đẩy bài sang người review, không xoá** — chặn nhầm tin thật tốn chi phí nhân sự thật, còn lọt spam vẫn còn các lớp lọc khác. Nhưng con số cuối do cả nhóm chốt sau khi nhìn bảng, và bảng đó được commit vào repo làm bằng chứng cho phần Responsible AI.

Bước này cần chạy lại inference một lần để lấy logits (~24.000 mẫu: 12k val ID + 7.6k test ID + ~4.5k OOD). Logits được cache ra `.npy` để các lần sweep sau không phải chạy lại.

### D6 — Preprocessing tách đôi theo nhánh model

```
clean_text_tfidf()   → chỉ nhánh baseline   (lower, xoá URL/số/ký tự đặc biệt)
passthrough()        → chỉ nhánh RoBERTa    (text thô, đúng như lúc train)
```

Sửa train/serve skew ở cell 62. Phải sửa **trước** khi recalibrate, vì skew làm lệch phân phối logits ⇒ ngưỡng calibrate xong sẽ sai khi lên API.

Khoá bằng test: `tests/unit/test_preprocess.py` khẳng định nhánh RoBERTa không bao giờ gọi `clean_text_tfidf`.

### D7 — Tên nhãn lấy từ `ood_config.json`, không lấy từ `config.json`

`roberta_final/config.json` có `id2label = {0: "LABEL_0", ... 3: "LABEL_3"}`. Dùng nó thì API trả `LABEL_2` cho user. Tên thật (`World`, `Sports`, `Business`, `Sci/Tech`) chỉ nằm trong `ood_config.json`.

Đây là loại lỗi không crash, không log, không test nào bắt được trừ khi viết contract test riêng ⇒ `tests/unit/test_label_mapping.py` là bắt buộc, không optional.

### D8 — `max_len` là contract, đọc từ config

`tokenizer_config.json` ghi `model_max_length: 512`, nhưng model train ở 128 và **ngưỡng OOD được calibrate ở 128**. Serve ở 512 làm đổi phân phối logits ⇒ ngưỡng sai. Luôn đọc `max_len` từ `ood_config.json`, không hardcode, và test khẳng định giá trị == 128.

### D9 — Nạp model: MLflow Registry là chính, bake-in là fallback

`MODEL_SOURCE=registry|local` (env var).

*Alternatives:* chỉ mount volume (image nhẹ, nhưng người chấm phải tự tải model → rủi ro demo hỏng); chỉ bake vào image (chắc ăn, nhưng không thể hiện Model Registry).

*Lý do chọn cả hai:* `registry` thể hiện đúng MLOps và là đường mặc định; `local` đảm bảo demo vẫn chạy nếu MLflow container chết. Image cuối ~1.5 GB (torch CPU-only wheel + 498 MB model) — chấp nhận được, rubric không chấm kích thước image.

### D10 — MLflow: backfill RoBERTa, train live baseline

- **RoBERTa**: `scripts/backfill_mlflow.py` đọc `roberta_checkpoints/*/trainer_state.json` (có log macro-F1 từng epoch, learning rate, loss — dữ liệu thật, không bịa) + `baseline_results.json` + `ood_comparison.json` → tạo run với đúng params/metrics/artifacts.
- **Baseline**: train lại thật với `mlflow.autolog()`, ~1 phút.

*Lý do:* train lại RoBERTa tốn 2–3h GPU và có rủi ro ra số khác với báo cáo đã viết. Backfill dùng dữ liệu có thật đã ghi lại. Baseline train live để chứng minh pipeline hoạt động khi người chấm chạy `python -m aegis.models.train_baseline`.

### D11 — CI dùng tiny-model fixture; model validation tách workflow riêng

`conftest.py` cung cấp fixture `hf-internal-testing/tiny-random-RobertaForSequenceClassification` (~2 MB) cho toàn bộ unit + integration test. Predictor được inject qua `Depends()` nên mock được.

`tests/model/` (F1 floor, OOD recall/FPR, latency budget, golden regression) chạy ở `.github/workflows/model-validation.yml` — nightly + `workflow_dispatch`, không chạy ở mỗi PR.

*Lý do:* runner free 7 GB RAM / 14 GB disk. Tải 498 MB + torch mỗi PR sẽ chậm và mong manh; một lần timeout là cả pipeline đỏ.

### D12 — Thứ tự dựng: contract → pure functions → API + mock → Docker → real model → MLflow

```
1  Contract      config.py, schemas.py (có sẵn field ood Optional), predictor interface
2  Pure fn       preprocess (tách đôi), ood/scoring, label mapping   ← khoá D6, D7, D8
3  API + Mock    FastAPI đủ endpoint + metrics, MockPredictor        ← toàn bộ D2 xong
4  Docker        multi-stage, compose 4 service, Prometheus, Grafana ← rủi ro cao nhất, làm sớm
5  Real          RoBERTa + LogReg thật, recalibrate OOD              ← test model chạy lần đầu
6  MLflow        backfill + train baseline live + Registry
```

*Lý do:* bước 1–4 không cần GPU, không cần model 498 MB, không cần internet ổn định. 4/5 thành viên làm song song ngay từ giờ đầu, và CI xanh với coverage thật từ ngày đầu thay vì ngày cuối. Bước 4 (Docker/compose networking) là rủi ro cao nhất nên được đẩy lên sớm thay vì để cuối.

### D13 — Trang demo là một file HTML tĩnh phục vụ từ container `api`

FastAPI mount `StaticFiles` tại `/`, phục vụ một trang HTML dùng vanilla JS `fetch` gọi chính API đó. `/docs`, `/metrics` và mọi route `/v1/*` giữ nguyên.

*Alternatives:*
- **Không frontend, demo bằng Swagger UI + Grafana**: 0 giờ công, đúng kịch bản trong báo cáo tư vấn. Nhưng demo trông như developer tool, khó kể được câu chuyện nghiệp vụ trong 4 phút.
- **Streamlit thành service thứ 5**: Python thuần, nhóm quen. Nhưng thêm ~300MB container, thêm một điểm chết lúc demo, và trùng vai với Grafana ở phần biểu đồ.
- **React/Vite + nginx container**: đẹp nhất nhưng kéo Node vào CI, thêm build stage, thêm bề mặt cần Trivy scan — mà không đổi lấy điểm rubric nào.

*Lý do chọn:* frontend **không cộng điểm ở rubric phát triển** — soi toàn bộ mục 3.1 không có tiêu chí nào nhắc tới giao diện. Nó chỉ phục vụ 15% Live Demo của phần thuyết trình. Vì vậy giải pháp đúng là cái rẻ nhất đạt được mục tiêu đó: không service mới, không dependency runtime mới, không build step, không job CI mới, không sửa `docker-compose.yml`. Nếu deadline siết, xoá một file HTML là xong, phần còn lại của hệ thống không hề hay biết.

Trang này MUST là client thuần của API công khai — không có logic phân loại hay ngưỡng OOD nào nằm trong JavaScript. Mọi thứ nó hiển thị đều lấy từ `/v1/predict` và `/v1/model/info`, nên nó không thể trôi lệch khỏi hành vi thật của hệ thống.

## Risks / Trade-offs

**[Siết FPR xuống 5% kéo OOD recall từ ~89% xuống ~45%]** (ước lượng binormal từ AUROC 0.859) → Không hardcode ngưỡng; xuất operating-point table đầy đủ, cả nhóm chốt sau khi nhìn số thật. Commit bảng vào repo — nó trở thành bằng chứng cho Responsible AI thay vì lỗ hổng bị hỏi trong Q&A.

**[OOD detector có thể đang học "văn phong ngắn/informal" thay vì "ngoài miền"]** — proxy là `sms_spam` + `tweet_eval/hate`, cả hai đều ngắn, viết thường, teencode; AG News thì formal → Viết test đối chứng: văn bản dài, formal, nhưng sai domain (công thức nấu ăn, đoạn hợp đồng). Nếu detector trượt, báo cáo trung thực trong phần Responsible AI — đó là điểm cộng, không phải điểm trừ.

**[Docker image ~1.5 GB, build chậm, CI có thể timeout]** → Buildx layer cache trên GHA; torch CPU-only wheel; `.dockerignore` loại `roberta_checkpoints/`, `*.zip`, `notebooks/`; docker build tách job riêng chỉ chạy sau khi test xanh.

**[`transformers==5.15.0` là version rất mới, dễ vỡ khi resolve dependency]** → Pin cứng toàn bộ `requirements.txt`; test load model offline trong container (không có mạng) như một integration test.

**[`roberta_final/` thiếu `merges.txt` và `vocab.json`, chỉ có `tokenizer.json`]** → Fast tokenizer hoạt động bình thường, nhưng `use_fast=False` sẽ vỡ. Cấm dùng slow tokenizer; test cold-start load offline để bắt sớm.

**[5 người commit dồn trong thời gian hẹp → merge conflict]** → Cấm push thẳng `main`; PR bắt buộc CI xanh; `.pre-commit-config.yaml` có `check-added-large-files` để chặn ai đó lỡ commit 498 MB.

**[Grafana dashboard chỉ nằm trong volume, người chấm `docker compose up` thấy trống]** → Dashboard JSON + datasource YAML provisioned as code, commit vào repo, mount vào container. Có integration test kiểm tra file provisioning tồn tại.

**[`NullOODDetector` trở thành vĩnh viễn]** → `tests/unit/test_ood_interface.py` parametrize qua mọi implementation; `/v1/model/info` phơi `ood_enabled` ra ngoài và Grafana có panel hiển thị trạng thái đó.

## Migration Plan

Đây là greenfield trên artifacts có sẵn — không có hệ thống đang chạy để migrate. "Migration" ở đây là thứ tự bật tính năng và đường lùi:

| Bước | Bật | Đường lùi nếu hỏng |
|---|---|---|
| 1 | API + MockPredictor | — |
| 2 | `MODEL_SOURCE=local` (bake-in) | — |
| 3 | `MODEL_SOURCE=registry` (MLflow) | về `local`, demo vẫn chạy |
| 4 | `OOD_ENABLED=true` | về `false`, `ood: null`, schema không đổi |
| 5 | Ngưỡng OOD recalibrated | `ood_config.json` cũ vẫn giữ trong git history |

Nếu compose networking bế tắc quá lâu: tháo MLflow khỏi compose, chuyển sang file-based tracking URI, giữ 3 service còn lại. Mất một phần điểm kiến trúc, giữ được demo.

## Open Questions

1. **Ngưỡng FPR cuối cùng** — 5% là mặc định đề xuất, nhưng chốt thật sau khi nhóm xem `ood_operating_points.json`. Nếu recall ở 5% thấp hơn ~40%, cân nhắc 10% hoặc chuyển sang hai ngưỡng (soft → review, hard → reject).
2. **Ai chạy inference lấy logits** — cần ~30 phút GPU Colab hoặc ~45 phút CPU. Chưa xác nhận ai còn session.
3. **Đã có ai bắt đầu code OOD riêng chưa** — nếu có, phải hợp nhất với notebook Phase 4 trước khi implement, tránh hai phiên bản lệch nhau.
4. **Nhóm duyệt sửa `MLOps.docx` mục Scope chưa** (D2) — đây là tài liệu chung, không phải quyết định của một người.
5. **`/explain` dùng LIME hay SHAP** — LIME nhanh hơn trên transformer nhưng kém ổn định; ảnh hưởng mục E (10%), chưa nằm trong scope change này nhưng API contract phải chừa chỗ.
