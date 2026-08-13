"""Task 8.4 — recreates an MLflow run for RoBERTa from
content/aegis_artifacts/roberta_checkpoints/*/trainer_state.json.

Uses only numbers HuggingFace's Trainer actually logged during the real
training run in notebooks/aegis_ag_news_training.ipynb — this is a backfill
of real history into MLflow, not a re-run or a fabrication (design.md D10).
Tagged source=backfill so it's visibly distinct from a live run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHECKPOINT_DIR = Path("content/aegis_artifacts/roberta_checkpoints/checkpoint-13500")
MODEL_COMPARISON_PATH = Path("content/aegis_artifacts/model_comparison.json")


def _load_test_macro_f1() -> float | None:
    if not MODEL_COMPARISON_PATH.exists():
        return None
    rows = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    row = next((r for r in rows if r["model"] == "RoBERTa-base"), None)
    return float(row["test_macro_f1"]) if row else None


def backfill(tracking_uri: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> str:
    import mlflow

    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("aegis-roberta")

    with mlflow.start_run(run_name="roberta-base-backfill") as run:
        mlflow.set_tags(
            {
                "source": "backfill",
                "source_checkpoint": str(checkpoint_dir),
                "note": "Backfilled from a real training run's trainer_state.json — see notebooks/aegis_ag_news_training.ipynb",
            }
        )

        mlflow.log_params(
            {
                "model_name": "roberta-base",
                "max_len": 128,
                "train_batch_size": state["train_batch_size"],
                "num_train_epochs": state["num_train_epochs"],
                "logging_steps": state["logging_steps"],
                "max_steps": state["max_steps"],
                "early_stopping_patience": state["stateful_callbacks"]["EarlyStoppingCallback"]["args"][
                    "early_stopping_patience"
                ],
                "seed": 42,
            }
        )

        for entry in state["log_history"]:
            step = entry["step"]
            if "loss" in entry:
                mlflow.log_metric("train_loss", entry["loss"], step=step)
                mlflow.log_metric("learning_rate", entry["learning_rate"], step=step)
            if "eval_macro_f1" in entry:
                mlflow.log_metric("eval_macro_f1", entry["eval_macro_f1"], step=step)
                mlflow.log_metric("eval_accuracy", entry["eval_accuracy"], step=step)
                mlflow.log_metric("eval_loss", entry["eval_loss"], step=step)

        mlflow.log_metric("best_val_macro_f1", state["best_metric"])
        mlflow.log_param("best_global_step", state["best_global_step"])

        test_macro_f1 = _load_test_macro_f1()
        if test_macro_f1 is not None:
            mlflow.log_metric("test_macro_f1", test_macro_f1)

        mlflow.log_artifact(str(checkpoint_dir / "trainer_state.json"))

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://localhost:5001")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    run_id = backfill(args.tracking_uri, args.checkpoint_dir)
    print(f"Backfilled MLflow run: {run_id}")


if __name__ == "__main__":
    main()
