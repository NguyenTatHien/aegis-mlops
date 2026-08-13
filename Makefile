.PHONY: up down logs test test-unit test-integration test-data test-model lint fmt train

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -m "unit or integration" --cov=src/aegis --cov-report=term-missing

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-data:
	pytest -m data

test-model:
	pytest -m model

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

fmt:
	ruff check --fix src tests
	ruff format src tests

train:
	python -m aegis.models.train_baseline
