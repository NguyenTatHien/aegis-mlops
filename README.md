# Aegis — News Routing & Out-of-Domain Detection

Classifies news text into `World / Sports / Business / Sci-Tech` (AG News) and flags text that falls outside that domain (spam, ads, off-topic content) so it can be routed to a human reviewer instead of being force-classified.

Three trained model branches sit behind one API: `roberta` (macro-F1 0.9517, default), `baseline` TF-IDF+LogisticRegression (0.9249), and the team's original `svm` TF-IDF+LinearSVC (0.9259). The demo and Grafana compare all three live. SVM exposes a relative decision margin and intentionally has OOD disabled because it has no calibrated probabilities.

## Quickstart

### 1. Prepare artifacts

Model artifacts live under `content/aegis_artifacts/` and are **not** committed to Git (too large — see `.gitignore`). A runnable artifact bundle must contain:

```text
content/aegis_artifacts/
├── roberta_final/{config.json,model.safetensors,tokenizer.json,tokenizer_config.json}
├── baseline/{logreg_tfidf_vectorizer.joblib,logreg_model.joblib,svm_tfidf_vectorizer.joblib,svm_model.joblib,baseline_results.json,svm_results.json}
├── ood_config.json
└── model_comparison.json
```

If you don't already have them:

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m aegis.models.train_baseline   # trains TF-IDF + LogisticRegression, ~1-2 min CPU
# RoBERTa: see notebooks/aegis_ag_news_training.ipynb (GPU recommended) or obtain
# roberta_final/ from the team's artifact store.
```

### 2. Run the full stack

```bash
cp .env.example .env
docker compose up -d --build
```

| Service | URL | What it is |
|---|---|---|
| API + demo page | http://localhost:8000/ | classify text, compare all three models |
| Swagger UI | http://localhost:8000/docs | interactive API docs |
| MLflow | http://localhost:5001 | experiment tracking, model registry |
| Prometheus | http://localhost:9090 | metrics, alert status |
| Grafana | http://localhost:3000 | dashboards (admin/admin) |

`docker compose ps` should show all four containers `healthy` within ~30s.

### 3. Try it

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The national football team won the championship last night."}'
```

Or just open http://localhost:8000/ — the demo page has sample buttons and a side-by-side model comparison.

## Development

```bash
pip install -r requirements-dev.txt
pre-commit install

make test          # unit + integration, coverage gate 80%
make lint          # ruff + mypy
make train         # retrain baseline
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for team roles, branch/commit conventions, and how to change OOD thresholds safely.

## Testing

Four kinds, matched to the course rubric:

| Type | Command | What it needs |
|---|---|---|
| Unit | `pytest -m unit` | nothing — pure functions, `MockPredictor` |
| Integration | `pytest -m integration` | nothing — FastAPI `TestClient` + `MockPredictor` |
| Data quality | `pytest -m data` | network (AG News download, cached after first run) |
| Model validation | `pytest -m model` | real artifacts under `content/aegis_artifacts/` |

`pytest -m "unit or integration"` is what CI gates on (coverage ≥ 80%); `data` and `model` run separately (nightly / on demand) since they need network or the full 498MB model — see `.github/workflows/`.

## Troubleshooting

- **`docker compose up` fails on `api` healthcheck** — check `docker compose logs api`. Most common cause: `content/aegis_artifacts/` isn't populated (see step 1) and `MODEL_SOURCE=local` can't find the files. Set `MODEL_SOURCE=mock` in `.env` to bring the stack up without real artifacts (useful for testing Docker/monitoring changes in isolation).
- **`GET /v1/predict` returns 503** — model still loading (RoBERTa takes a few seconds on first startup) or failed to load; check `docker compose logs api` for the startup exception.
- **Grafana shows no data** — confirm Prometheus target is up at http://localhost:9090/targets; confirm you've sent at least one request to `/v1/predict` (metrics only populate on traffic).
- **`pytest -m data` is slow the first time** — it downloads AG News (~30MB) via the `datasets` library; subsequent runs use the local HF cache.

## Known limitations

- OOD detection is calibrated against proxy datasets (`sms_spam`, `tweet_eval/hate`), not real production traffic — both proxies are short, informal text, while AG News is formal prose. `tests/model/test_ood_style_vs_domain.py` checks whether long, *formal* off-domain text (a recipe, a legal clause) is still caught; see that test's output for the current answer.
- Recalibrating the OOD threshold trades recall for false-positive rate — see `content/aegis_artifacts/ood_operating_points.json` for the full sweep and `ARCHITECTURE.md` for the reasoning behind the chosen operating point.
- LinearSVC has no `predict_proba`; the demo labels its softmax-normalized decision score as `relative margin`, records it in a separate Prometheus metric, and disables OOD for `model=svm`.
- LogReg and SVM MUST use their own fitted TF-IDF vectorizers. Their matrices both have 50,000 columns but different vocabulary ordering, so the vectorizers are never interchangeable.
- `/v1/explain` (LIME/SHAP token attribution) is a reserved endpoint, not yet implemented.
