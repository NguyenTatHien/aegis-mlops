# Architecture

## System diagram

```
                        ┌────────────────────────────────────────┐
                        │  Browser                                │
                        │  /            → demo page (static HTML)│
                        │  /docs        → Swagger UI              │
                        └───────────────┬──────────────────────────┘
                                        │
                        ┌───────────────▼──────────────────────────┐
                        │  container: api  (FastAPI + Uvicorn)      │
                        │  ├─ StaticFiles         → "/"             │
                        │  ├─ POST /v1/predict    (?model=)         │
                        │  ├─ POST /v1/predict/batch                │
                        │  ├─ GET  /v1/model/info                   │
                        │  ├─ GET  /health , /ready                 │
                        │  ├─ GET  /metrics       (Prometheus)      │
                        │  ├─ RobertaPredictor    (torch, CPU)      │
                        │  ├─ BaselinePredictor   (scikit-learn)    │
                        │  └─ OODDetector (MSP / Energy / Entropy)  │
                        └───┬───────────────┬────────────────┬──────┘
                            │               │                │
                 (nạp model)│    (ghi lại)  │     (thu thập) │
                            ▼               ▼                ▼
                    ┌──────────────┐ ┌─────────────┐ ┌───────────────┐
                    │ Model artifacts│ │ MLflow     │ │ Prometheus    │
                    │ (volume mount) │ │ (registry) │ │ (scrape 5s)   │
                    └──────────────┘ └─────────────┘ └───────┬───────┘
                                                              │
                                                       (truy vấn PromQL)
                                                              ▼
                                                       ┌───────────────┐
                                                       │ Grafana       │
                                                       │ (dashboard,   │
                                                       │  alerting)    │
                                                       └───────────────┘
```

Four containers (`api`, `mlflow`, `prometheus`, `grafana`) on one Docker network, orchestrated by `docker-compose.yml`. `api` waits for `mlflow` to report healthy before starting.

## Data flow

```
AG News (HuggingFace) ──┬─→ 90/10 stratified split (seed=42) ─┬─→ TF-IDF + LogisticRegression (baseline)
                         │                                     ├─→ TF-IDF + LinearSVC (original SVM)
                         │                                     └─→ RoBERTa-base fine-tune (offline, notebooks/)
                         │
sms_spam + tweet_eval/hate ─→ OOD proxy (spam / toxic) ─→ RoBERTa logits ─→ MSP / Energy scoring ─→ threshold sweep
                                                                                                     (scripts/recalibrate_ood.py)
```

Client request → `POST /v1/predict` → preprocessing branch (`clean_text_tfidf` for LogReg/SVM, `passthrough` for RoBERTa — never swapped, see Decisions) → predictor → `{predicted_class, confidence, ood}` → Prometheus metrics recorded → response.

## Key decisions and trade-offs

**Three trained models behind one API.** `/v1/predict?model=baseline|svm|roberta` defaults to `roberta`. The `svm` branch preserves the team's original `LinearSVC` artifact for live F1/label/latency comparison; its softmax-normalized decision score is explicitly a relative margin, not a calibrated probability, so OOD stays disabled for SVM. LogReg supplies `predict_proba` for the lightweight probabilistic branch, while RoBERTa remains the primary model and supports Energy OOD. Grafana splits probabilities and SVM relative margins into separate panels.

**OOD is a pure function of logits.** `OODDetector.score(logits) -> float` never tokenizes, loads a model, or reads a file. This is what let ~90% of the system (API, Docker, Prometheus, Grafana, CI) get built and tested with `MockPredictor` before the real 498MB model ever needed to load — see `openspec/changes/build-aegis-mlops-system/design.md` decision D12 for the full build order.

**Preprocessing follows the trained branch.** `clean_text_tfidf()` (lowercase, strip digits/punctuation) is shared by LogReg and LinearSVC because both were trained on the same TF-IDF representation. RoBERTa receives raw text; feeding it cleaned text would silently shift the logit distribution the OOD thresholds were calibrated against. This exact bug existed in the original research notebook's `aegis_predict()` helper and is now a regression test (`tests/model/test_real_predictors.py::test_roberta_predictor_text_not_cleaned_before_tokenize`).

**Each classical model owns its fitted vectorizer.** `logreg_tfidf_vectorizer.joblib` and `svm_tfidf_vectorizer.joblib` both expose 50,000 columns, but their vocabulary indices differ because they were fitted in separate training runs. Pairing SVM weights with the LogReg vectorizer silently produces plausible-shaped but invalid predictions, so serving and training use model-specific artifact names.

**`max_len=128` and label names come from `ood_config.json`, never from the model's own `config.json`.** `roberta_final/config.json` ships `id2label={"0":"LABEL_0",...}` and `model_max_length=512` — neither is correct for serving (real names are `World/Sports/Business/Sci-Tech`; the model was trained and OOD-calibrated at 128, not 512).

**OOD thresholds are chosen from a full FPR sweep, not hand-picked.** The original notebook's thresholds gave FPR 35-41% (roughly a third of legitimate news gets flagged) because it optimized for recall first. `scripts/recalibrate_ood.py` sweeps FPR targets {1,2,5,10,15,20,30}% for both MSP and Energy and writes the whole table to `ood_operating_points.json` before anyone commits to one number — see that file for what the team actually chose and why.

## Why RoBERTa is CPU-only in this deployment

No GPU in the target environment; `torch` is installed from `download.pytorch.org/whl/cpu` specifically to avoid pulling in the ~2GB CUDA runtime. Latency budget (p95 < 500ms, see `tests/model/test_latency.py`) is met on CPU for single-request inference; batch endpoint exists but isn't optimized for GPU-scale throughput — out of scope per `design.md` Non-Goals.
