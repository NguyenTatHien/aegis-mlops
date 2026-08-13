# Contributing

## Roles

| Member | Area | Owns |
|---|---|---|
| **M1** — Data / ML Engineer | Data pipeline, model training, OOD calibration | `src/aegis/data/`, `src/aegis/models/`, `src/aegis/ood/`, `scripts/recalibrate_ood.py`, `notebooks/` |
| **M2** — MLOps / Tracking Lead | Experiment tracking, model registry | `scripts/backfill_mlflow.py`, MLflow integration in `train_baseline.py`, `docker-compose.yml`'s `mlflow` service |
| **M3** — Backend / API Architect | Serving layer, demo page | `src/aegis/api/`, `src/aegis/serving/` |
| **M4** — DevOps / CI-CD Engineer | Containers, pipelines | `docker/`, `docker-compose.yml`, `.github/workflows/`, `Makefile` |
| **M5** — QA / Monitoring / Responsible AI Lead | Tests, dashboards, alerts, fairness | `tests/`, `monitoring/`, `README.md` |

## Branching and commits

- No direct pushes to `main`. Every change goes through a pull request.
- CI (`ci.yml`) must be green before merge: lint → unit/integration tests (coverage ≥ 80%) → data tests → Docker build + Trivy scan → compose smoke test.
- Use conventional, meaningful commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Every team member should have commits reflecting the area they own — this maps to the individual contribution grading in the course rubric.

## Local setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows cmd
pip install -r requirements-dev.txt
pre-commit install
```

Run the fast suite before pushing:

```bash
pytest -m "unit or integration" --cov=src/aegis --cov-report=term-missing
ruff check src tests && ruff format --check src tests && mypy src
```

## Changing the OOD threshold

Never hand-edit `content/aegis_artifacts/ood_config.json`. Run:

```bash
python scripts/collect_logits.py      # only needed if the model changed
python scripts/recalibrate_ood.py --target-fpr 0.05
```

Review `content/aegis_artifacts/ood_operating_points.json` before committing a new `ood_config.json` — it's the full FPR sweep table, not just the one value that got picked.

## Where things live

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system diagram and design rationale, and `openspec/changes/build-aegis-mlops-system/` for the full proposal/design/spec/tasks this build followed.
