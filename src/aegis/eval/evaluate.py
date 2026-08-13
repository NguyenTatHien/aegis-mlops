from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], output_dir: Path, prefix: str
) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred)
    _save_confusion_matrix_png(
        cm, label_names, output_dir / f"{prefix}_confusion_matrix.png", prefix
    )

    return {"macro_f1": macro_f1, "classification_report": report}


def _save_confusion_matrix_png(
    cm: np.ndarray, label_names: list[str], path: Path, title_prefix: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
