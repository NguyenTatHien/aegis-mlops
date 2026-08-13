"""Task 7.5 — sweeps FPR targets {1,2,5,10,15,20,30}% for MSP and Energy
using cached logits from scripts/collect_logits.py. Writes:
  - content/aegis_artifacts/ood_operating_points.json  (full sweep table)
  - content/aegis_artifacts/ood_roc_curve.png           (MSP vs Energy, test set)
  - content/aegis_artifacts/ood_config.json             (new, at --target-fpr)

Old ood_config.json (FPR 35-41%) is not overwritten in place without being
diffed first — this script prints old vs new before writing (design.md D5:
"đừng hardcode một con số... nhóm nhìn số thật mà chọn").

Thresholds are chosen on the VALIDATION split, then measured_fpr/
measured_recall in the output are evaluated on the held-out TEST split —
the number that matters is generalization, not the number that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.config import get_label_names, get_max_len  # noqa: E402
from aegis.ood.scoring import energy_score, msp_score  # noqa: E402

TARGET_FPRS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
LOGITS_DIR = Path("content/aegis_artifacts/ood_logits")
OOD_CONFIG_PATH = Path("content/aegis_artifacts/ood_config.json")
OPERATING_POINTS_PATH = Path("content/aegis_artifacts/ood_operating_points.json")
ROC_PLOT_PATH = Path("content/aegis_artifacts/ood_roc_curve.png")


def _roc_sweep(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, thresholds = roc_curve(y, scores)
    auroc = float(roc_auc_score(y, scores))
    return fpr, tpr, thresholds, auroc


def _threshold_at_fpr(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, target: float) -> tuple[float, int]:
    valid_idx = np.where(fpr <= target)[0]
    idx = int(valid_idx[np.argmax(tpr[valid_idx])]) if len(valid_idx) else int(np.argmin(fpr))
    return float(thresholds[idx]), idx


def _apply_threshold(id_scores: np.ndarray, ood_scores: np.ndarray, threshold: float) -> tuple[float, float]:
    fpr = float((id_scores >= threshold).mean())
    recall = float((ood_scores >= threshold).mean())
    return fpr, recall


def sweep(energy_temperature: float = 1.0) -> dict:
    id_val = np.load(LOGITS_DIR / "id_val_logits.npy")
    ood_val = np.load(LOGITS_DIR / "ood_val_logits.npy")
    id_test = np.load(LOGITS_DIR / "id_test_logits.npy")
    ood_test = np.load(LOGITS_DIR / "ood_test_logits.npy")

    points = []
    per_method_test_scores = {}

    for method, score_fn in (
        ("msp", msp_score),
        ("energy", lambda logits: energy_score(logits, energy_temperature)),
    ):
        val_id_scores = score_fn(id_val)
        val_ood_scores = score_fn(ood_val)
        test_id_scores = score_fn(id_test)
        test_ood_scores = score_fn(ood_test)
        per_method_test_scores[method] = (test_id_scores, test_ood_scores)

        fpr_curve, tpr_curve, thresholds, val_auroc = _roc_sweep(val_id_scores, val_ood_scores)
        _, _, _, test_auroc = _roc_sweep(test_id_scores, test_ood_scores)

        for target in TARGET_FPRS:
            threshold, idx = _threshold_at_fpr(fpr_curve, tpr_curve, thresholds, target)
            test_fpr, test_recall = _apply_threshold(test_id_scores, test_ood_scores, threshold)
            points.append(
                {
                    "method": method,
                    "target_fpr": target,
                    "threshold": threshold,
                    "val_fpr": float(fpr_curve[idx]),
                    "val_recall": float(tpr_curve[idx]),
                    "measured_fpr": test_fpr,
                    "measured_recall": test_recall,
                    "val_auroc": val_auroc,
                    "test_auroc": test_auroc,
                }
            )

    return {"operating_points": points, "test_scores": per_method_test_scores}


def plot_roc(test_scores: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    for method, (id_scores, ood_scores) in test_scores.items():
        y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=method)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate (ID flagged as OOD)")
    ax.set_ylabel("True Positive Rate (OOD Recall)")
    ax.set_title("MSP vs Energy — ROC on held-out OOD test set")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_operating_point(operating_points: list[dict], method: str, target_fpr: float) -> dict:
    return next(p for p in operating_points if p["method"] == method and p["target_fpr"] == target_fpr)


def build_ood_config(
    operating_points: list[dict],
    target_fpr: float,
    energy_temperature: float = 1.0,
    label_names: list[str] | None = None,
    max_len: int | None = None,
) -> dict:
    msp_point = select_operating_point(operating_points, "msp", target_fpr)
    energy_point = select_operating_point(operating_points, "energy", target_fpr)
    return {
        "msp_threshold": msp_point["threshold"],
        "energy_threshold": energy_point["threshold"],
        "energy_temperature": energy_temperature,
        "label_names": label_names if label_names is not None else get_label_names(),
        "max_len": max_len if max_len is not None else get_max_len(),
        "target_fpr": target_fpr,
        "measured_fpr": {"msp": msp_point["measured_fpr"], "energy": energy_point["measured_fpr"]},
        "measured_recall": {"msp": msp_point["measured_recall"], "energy": energy_point["measured_recall"]},
        "calibrated_at": datetime.now(UTC).isoformat(),
    }


def log_to_mlflow(tracking_uri: str, new_config: dict, operating_points_path: Path, roc_plot_path: Path) -> str:
    """Task 8.5. Logs the chosen operating point as metrics/params and the
    full sweep table + ROC plot as artifacts — a reviewer should be able to
    see *why* this FPR/threshold was picked, not just the final numbers."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("aegis-ood-calibration")

    with mlflow.start_run(run_name="ood-recalibration") as run:
        mlflow.log_param("target_fpr", new_config["target_fpr"])
        mlflow.log_param("energy_temperature", new_config["energy_temperature"])
        mlflow.log_metric("msp_threshold", new_config["msp_threshold"])
        mlflow.log_metric("energy_threshold", new_config["energy_threshold"])
        for method, fpr in new_config["measured_fpr"].items():
            mlflow.log_metric(f"{method}_measured_fpr", fpr)
        for method, recall in new_config["measured_recall"].items():
            mlflow.log_metric(f"{method}_measured_recall", recall)
        mlflow.log_artifact(str(operating_points_path))
        mlflow.log_artifact(str(roc_plot_path))
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Default operating point (team decision)")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="If set, logs the result to MLflow (task 8.5)")
    args = parser.parse_args()

    print("Sweeping FPR targets for MSP and Energy using cached logits...")
    result = sweep()
    operating_points = result["operating_points"]

    OPERATING_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPERATING_POINTS_PATH.write_text(
        json.dumps([{k: v for k, v in p.items()} for p in operating_points], indent=2), encoding="utf-8"
    )
    print(f"Wrote {OPERATING_POINTS_PATH}")

    plot_roc(result["test_scores"], ROC_PLOT_PATH)
    print(f"Wrote {ROC_PLOT_PATH}")

    print("\n--- Operating point table (test-set FPR/recall) ---")
    print(f"{'method':<8} {'target_fpr':<12} {'threshold':<12} {'measured_fpr':<14} {'measured_recall':<16} test_auroc")
    for p in operating_points:
        print(
            f"{p['method']:<8} {p['target_fpr']:<12} {p['threshold']:<12.4f} "
            f"{p['measured_fpr']:<14.4f} {p['measured_recall']:<16.4f} {p['test_auroc']:.4f}"
        )

    old_config = json.loads(OOD_CONFIG_PATH.read_text(encoding="utf-8")) if OOD_CONFIG_PATH.exists() else None
    new_config = build_ood_config(operating_points, target_fpr=args.target_fpr)

    print(f"\n--- ood_config.json: old vs new (target_fpr={args.target_fpr}) ---")
    print("old:", json.dumps(old_config, indent=2) if old_config else "(none)")
    print("new:", json.dumps(new_config, indent=2))

    OOD_CONFIG_PATH.write_text(json.dumps(new_config, indent=2), encoding="utf-8")
    print(f"\nWrote {OOD_CONFIG_PATH}")

    if args.mlflow_tracking_uri:
        run_id = log_to_mlflow(args.mlflow_tracking_uri, new_config, OPERATING_POINTS_PATH, ROC_PLOT_PATH)
        print(f"Logged MLflow run: {run_id}")


if __name__ == "__main__":
    main()
