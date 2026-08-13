## What changed

<!-- One or two sentences. -->

## Why

<!-- Link the task/issue if applicable. -->

## Checklist

- [ ] `pytest -m "unit or integration"` passes locally with coverage ≥ 80%
- [ ] `ruff check` / `ruff format --check` / `mypy src` pass locally
- [ ] No file over 10MB added (checked by pre-commit)
- [ ] Updated `README.md` / `ARCHITECTURE.md` if this changes setup or design
- [ ] If this touches OOD thresholds or `ood_config.json`: ran `scripts/recalibrate_ood.py` and reviewed `ood_operating_points.json`, not just edited the file by hand

## Test plan

<!-- How did you verify this? -->
