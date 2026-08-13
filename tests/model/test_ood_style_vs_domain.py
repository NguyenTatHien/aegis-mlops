"""Task 7.15 — is the OOD detector catching domain shift, or just short/
informal writing style? design.md risk: the proxy datasets (sms_spam,
tweet_eval/hate) are both short and informal; AG News is formal news prose.
This test feeds long, FORMAL, off-domain text (a recipe, a legal clause) and
records whether the detector catches it. If it doesn't, that's a documented
limitation for the Responsible AI writeup (task 11.6/11.8), not a bug to
silently patch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

ROBERTA_DIR = Path("content/aegis_artifacts/roberta_final")
OOD_CONFIG_PATH = Path("content/aegis_artifacts/ood_config.json")

FORMAL_OFF_DOMAIN_TEXTS = {
    "recipe": (
        "Preheat the oven to 180 degrees Celsius. In a large mixing bowl, cream the butter and sugar together "
        "until light and fluffy. Gradually add the eggs, one at a time, beating well after each addition. Sift "
        "the flour, baking powder, and salt together, then fold gently into the mixture. Pour the batter into a "
        "greased cake tin and bake for approximately forty-five minutes, or until a skewer inserted into the "
        "center comes out clean. Allow the cake to cool completely before removing it from the tin."
    ),
    "legal_clause": (
        "This Agreement shall be governed by and construed in accordance with the laws of the jurisdiction in "
        "which the parties are domiciled, without regard to its conflict of law provisions. Any dispute arising "
        "out of or in connection with this Agreement, including any question regarding its existence, validity, "
        "or termination, shall be referred to and finally resolved by arbitration administered in accordance "
        "with the applicable rules then in effect. Each party shall bear its own costs incurred in connection "
        "with such proceedings unless otherwise determined by the arbitral tribunal."
    ),
}


def _require_calibrated_setup():
    if not (ROBERTA_DIR / "model.safetensors").exists():
        pytest.skip("roberta_final artifacts not present")
    if not OOD_CONFIG_PATH.exists():
        pytest.skip("ood_config.json not present")
    config = json.loads(OOD_CONFIG_PATH.read_text(encoding="utf-8"))
    if "measured_fpr" not in config:
        pytest.skip("ood_config.json not recalibrated yet")
    return config


@pytest.mark.parametrize("case_name", list(FORMAL_OFF_DOMAIN_TEXTS))
def test_formal_off_domain_text_ood_outcome_is_recorded(case_name: str) -> None:
    """Not a pass/fail assertion on detection — a recorded observation.
    Prints the result so it shows up in CI logs for task 11.8's writeup."""
    config = _require_calibrated_setup()
    from aegis.ood.detector import EnergyDetector
    from aegis.serving.roberta_predictor import RobertaPredictor

    predictor = RobertaPredictor(ROBERTA_DIR)
    detector = EnergyDetector(
        threshold=config["energy_threshold"], temperature=config["energy_temperature"]
    )

    result = predictor.predict(FORMAL_OFF_DOMAIN_TEXTS[case_name])
    score = detector.score(result.logits)
    is_ood = detector.is_ood(score)

    print(
        f"\n[style-vs-domain] case={case_name} predicted={result.predicted_class} "
        f"confidence={result.confidence:.3f} energy_score={score:.3f} "
        f"threshold={config['energy_threshold']:.3f} flagged_ood={is_ood}"
    )
    # Intentionally no assertion on is_ood — see module docstring.
    assert isinstance(is_ood, bool)
