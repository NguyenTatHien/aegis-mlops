"""Task 7.4 — runs RoBERTa inference once over ID val/test and OOD val/test,
caching logits to .npy so scripts/recalibrate_ood.py can sweep thresholds
without re-running inference (design.md D5).

Text goes in raw (passthrough), never clean_text_tfidf() — this is the exact
place the train/serve skew bug (design.md D6) would resurface if someone
copied notebooks/aegis_ag_news_training.ipynb cell 62's aegis_predict()
instead of using aegis.data.preprocess.passthrough().
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.config import get_max_len  # noqa: E402
from aegis.data.loader import load_ood_proxy_texts, prepare_ag_news_splits  # noqa: E402
from aegis.data.preprocess import passthrough  # noqa: E402

MODEL_DIR = Path("content/aegis_artifacts/roberta_final")
OUTPUT_DIR = Path("content/aegis_artifacts/ood_logits")


def get_logits_batch(tokenizer, model, texts: list[str], max_len: int, batch_size: int = 32) -> np.ndarray:
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [passthrough(t) for t in texts[i : i + batch_size]]
            encoded = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            logits = model(**encoded).logits.numpy()
            all_logits.append(logits)
            if (i // batch_size) % 20 == 0:
                print(f"  {i + len(batch)}/{len(texts)}")
    return np.concatenate(all_logits, axis=0)


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_len = get_max_len()

    print(f"Loading RoBERTa from {MODEL_DIR} (max_len={max_len})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    print("Preparing AG News splits (seed=42, val_size=0.10)...")
    splits = prepare_ag_news_splits(seed=42, val_size=0.10)

    print("Loading OOD proxy datasets (sms_spam + tweet_eval/hate)...")
    ood_val_texts, ood_test_texts = load_ood_proxy_texts(seed=42)
    print(f"  OOD val: {len(ood_val_texts)}  OOD test: {len(ood_test_texts)}")

    jobs = {
        "id_val": splits["val_texts"],
        "id_test": splits["test_texts"],
        "ood_val": ood_val_texts,
        "ood_test": ood_test_texts,
    }

    for name, texts in jobs.items():
        t0 = time.time()
        print(f"Running inference: {name} ({len(texts)} texts)")
        logits = get_logits_batch(tokenizer, model, texts, max_len)
        out_path = OUTPUT_DIR / f"{name}_logits.npy"
        np.save(out_path, logits)
        print(f"  saved {logits.shape} -> {out_path} in {time.time() - t0:.1f}s")

    np.save(OUTPUT_DIR / "id_val_labels.npy", splits["val_labels"])
    np.save(OUTPUT_DIR / "id_test_labels.npy", splits["test_labels"])
    print(f"Done. Logits cached under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
