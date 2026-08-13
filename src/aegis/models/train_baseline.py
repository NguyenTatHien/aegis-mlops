"""TF-IDF + LogisticRegression baseline.

LogisticRegression, not the notebook's LinearSVC (design.md D3) — it has
predict_proba, which the API needs for `confidence` and which entropy-based
OOD (group 7) needs as its input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aegis.data.preprocess import clean_text_tfidf

DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def train_baseline(
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    label_names: list[str],
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import GridSearchCV

    cleaned_train = [clean_text_tfidf(t) for t in train_texts]
    cleaned_val = [clean_text_tfidf(t) for t in val_texts]
    cleaned_test = [clean_text_tfidf(t) for t in test_texts]

    vectorizer = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    )
    x_train = vectorizer.fit_transform(cleaned_train)
    x_val = vectorizer.transform(cleaned_val)
    x_test = vectorizer.transform(cleaned_test)

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=seed),
        {"C": list(c_grid)},
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(x_train, train_labels)
    best_c = float(grid.best_params_["C"])

    model = LogisticRegression(C=best_c, max_iter=1000, random_state=seed)
    model.fit(x_train, train_labels)

    val_pred = model.predict(x_val)
    val_macro_f1 = float(f1_score(val_labels, val_pred, average="macro"))

    test_pred = model.predict(x_test)
    test_macro_f1 = float(f1_score(test_labels, test_pred, average="macro"))

    return {
        "vectorizer": vectorizer,
        "model": model,
        "best_c": best_c,
        "val_macro_f1": val_macro_f1,
        "test_macro_f1": test_macro_f1,
        "cv_results": {
            "C": list(c_grid),
            "mean_test_score": [float(v) for v in grid.cv_results_["mean_test_score"]],
        },
        "test_classification_report": classification_report(
            test_labels, test_pred, target_names=label_names, output_dict=True
        ),
    }


def log_to_mlflow(
    result: dict[str, Any], tracking_uri: str, artifacts_dir: Path
) -> str:  # pragma: no cover
    """Logs one parent run + one child run per swept C value (task 8.2 —
    "mỗi giá trị C là một run"). Requires a reachable MLflow tracking server;
    kept out of train_baseline() itself so unit tests can call that function
    without mlflow ever being imported (design.md D10)."""
    import mlflow
    import mlflow.sklearn
    from sklearn.pipeline import Pipeline

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("aegis-baseline")

    with mlflow.start_run(run_name="baseline-tfidf-logreg") as run:
        mlflow.log_params({"max_features": 50_000, "ngram_range": "(1,2)", "seed": 42})

        for c, score in zip(
            result["cv_results"]["C"], result["cv_results"]["mean_test_score"], strict=True
        ):
            with mlflow.start_run(run_name=f"C={c}", nested=True):
                mlflow.log_param("C", c)
                mlflow.log_metric("cv_macro_f1", score)

        mlflow.log_param("best_C", result["best_c"])
        mlflow.log_metric("val_macro_f1", result["val_macro_f1"])
        mlflow.log_metric("test_macro_f1", result["test_macro_f1"])
        mlflow.log_artifacts(str(artifacts_dir))

        # Registered as vectorizer+classifier bundled together — a caller
        # loading models:/aegis-baseline/Production gets one object whose
        # .predict_proba(texts) works directly, not just the bare
        # classifier (which alone can't accept raw text).
        pipeline = Pipeline([("tfidf", result["vectorizer"]), ("clf", result["model"])])
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", registered_model_name="aegis-baseline"
        )

        return run.info.run_id


def save_baseline_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["vectorizer"], output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(result["model"], output_dir / "logreg_model.joblib")

    report = {k: v for k, v in result.items() if k not in {"vectorizer", "model"}}
    report["model"] = "TF-IDF + LogisticRegression"
    (output_dir / "baseline_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    import os

    from aegis.data.loader import prepare_ag_news_splits

    splits = prepare_ag_news_splits(seed=42, val_size=0.10)

    result = train_baseline(
        splits["train_texts"],
        splits["train_labels"],
        splits["val_texts"],
        splits["val_labels"],
        splits["test_texts"],
        splits["test_labels"],
        splits["label_names"],
    )
    print(
        f"best_C={result['best_c']} val_macro_f1={result['val_macro_f1']:.4f} test_macro_f1={result['test_macro_f1']:.4f}"
    )

    artifacts_dir = Path("content/aegis_artifacts/baseline")
    save_baseline_artifacts(result, artifacts_dir)
    print(f"Saved baseline artifacts to {artifacts_dir}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        run_id = log_to_mlflow(result, tracking_uri, artifacts_dir)
        print(f"Logged MLflow run: {run_id}")
    else:
        print("MLFLOW_TRACKING_URI not set — skipping MLflow logging")
