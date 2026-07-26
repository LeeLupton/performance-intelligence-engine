"""Real-data validation gate: the go/no-go step between a trained model and
production.

A trained checkpoint carries synthetic calibration and no defensible operating
threshold (see reports/PRODUCTION_READINESS.md). This module takes real labeled
campaign windows and, on a strictly time-ordered holdout, does the four things
that make a model deployable:

1. re-fit affine calibration on an earlier real segment (the "dev" segment);
2. select an operating threshold at a target false-positive rate on that segment;
3. snapshot a real-traffic drift baseline so production drift is measured
   against reality, not the simulator;
4. verify metrics, calibration, and the *realized* FPR at the chosen threshold
   on a later, untouched real segment (the "test" segment) — and turn a set of
   configurable gates into a single go/no-go verdict plus a model card.

Honesty is enforced structurally: the tool cannot tell real data from
simulator output, so the operator must attest the data provenance. A verdict is
only ``binding`` when that attestation names real data — running the gate on
the simulator as a wiring stand-in produces a full report whose verdict is
explicitly non-binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .models import CampaignModel
from .registry import feature_schema_hash
from .schema import LabeledWindow
from .training import (
    Batch,
    _evaluate,
    affine_calibration_params,
    feature_snapshot,
    windows_to_batch,
)

# Default gates — the operator's risk appetite, overridable. Each guards a
# distinct failure mode: too little data to trust the estimate, poor ranking,
# miscalibrated probabilities, too few true detections at the tolerated FPR, or
# a threshold that does not actually hold its FPR on unseen traffic.
DEFAULT_GATES: dict[str, float] = {
    "min_windows_per_segment": 20,
    "min_positives_per_segment": 5,
    "min_roc_auc": 0.75,
    "max_ece": 0.10,
    "min_recall_at_target_fpr": 0.50,
    "max_realized_fpr_ratio": 1.5,  # realized test FPR at tau <= target_fpr * this
}

# Provenance strings containing any of these read as "not real data", so the
# verdict is reported but never binding. Deliberately conservative — a false
# "non-binding" only asks the operator to rename; a false "binding" would
# rubber-stamp synthetic data as production. ("test" is intentionally absent:
# pentest/red-team captures are real data.)
_SYNTHETIC_MARKERS = ("unspecified", "synthetic", "sim", "simulator", "standin", "stand-in", "demo")


@dataclass(frozen=True)
class Threshold:
    """An operating point: the probability cutoff and what it achieved on dev."""

    tau: float
    dev_fpr: float
    dev_tpr: float


def threshold_at_fpr(labels: np.ndarray, probability: np.ndarray, target_fpr: float) -> Threshold:
    """Highest-recall probability cutoff whose FPR stays at or below target_fpr.

    Returns tau=inf (detect nothing) when no admissible operating point exists,
    which correctly fails the recall gate downstream rather than inventing one.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, probability)
    admissible = np.where(fpr <= target_fpr)[0]
    if admissible.size == 0:
        return Threshold(tau=float("inf"), dev_fpr=0.0, dev_tpr=0.0)
    best = admissible[int(np.argmax(tpr[admissible]))]
    # roc_curve's first threshold is an above-range sentinel (np.inf on modern
    # sklearn); keep it as-is so tau=inf means "detect nothing", consistent
    # with the no-admissible-point case.
    return Threshold(tau=float(thresholds[best]), dev_fpr=float(fpr[best]), dev_tpr=float(tpr[best]))


def _confusion_at(labels: np.ndarray, probability: np.ndarray, tau: float) -> dict[str, Any]:
    """Confusion counts and rates when alerting on probability >= tau."""
    predicted = probability >= tau
    positives = labels == 1
    tp = int(np.sum(predicted & positives))
    fp = int(np.sum(predicted & ~positives))
    tn = int(np.sum(~predicted & ~positives))
    fn = int(np.sum(~predicted & positives))
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "realized_fpr": round(fpr, 6),
        "realized_tpr": round(tpr, 6),
        "precision": round(precision, 6) if precision is not None else None,
    }


def _logits(model: CampaignModel, batch: Batch) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(batch.sequences, batch.mask, batch.adjacency, batch.deltas).graph_logit


def _segment_labels(batch: Batch) -> tuple[int, int]:
    positives = int(batch.labels.sum().item())
    return len(batch.labels), positives


def validate_model(
    model: CampaignModel,
    windows: list[LabeledWindow],
    target_fpr: float = 0.01,
    holdout: float = 0.5,
    provenance: str = "unspecified",
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the go/no-go gate over time-ordered labeled windows; return the report.

    The model is mutated in place to carry the real calibration and the real
    drift baseline, so a checkpoint saved after this call is the validated
    artifact. Windows must already be chronologically sortable (they carry a
    start time); they are split so the test segment is strictly later than the
    dev segment — no future data informs calibration or the threshold.
    """
    if not 0.1 <= holdout <= 0.9:
        raise ValueError("holdout must be between 0.1 and 0.9")
    if len(windows) < 4:
        raise ValueError("need at least 4 labeled windows to form two segments")
    unknown = set(gates or {}) - set(DEFAULT_GATES)
    if unknown:
        raise ValueError(f"unknown gate(s): {', '.join(sorted(unknown))}; valid gates are {', '.join(sorted(DEFAULT_GATES))}")
    gates = {**DEFAULT_GATES, **(gates or {})}
    ordered = sorted(windows, key=lambda window: (window.start, window.window_id))
    split = round(len(ordered) * (1.0 - holdout))
    split = min(max(split, 2), len(ordered) - 2)
    dev_windows, test_windows = ordered[:split], ordered[split:]

    dev = windows_to_batch(dev_windows, time_mode=model.time_mode)
    test = windows_to_batch(test_windows, time_mode=model.time_mode)
    dev_n, dev_pos = _segment_labels(dev)
    test_n, test_pos = _segment_labels(test)
    for name, batch in (("dev", dev), ("test", test)):
        classes = int(batch.labels.sum().item())
        if classes == 0 or classes == len(batch.labels):
            raise ValueError(
                f"{name} segment is single-class ({classes}/{len(batch.labels)} malicious); "
                "supply windows spanning both classes in each time segment"
            )

    # 1. Re-fit calibration on the dev segment's real logits.
    dev_logits = _logits(model, dev)
    scale, bias = affine_calibration_params(dev_logits, dev.labels)
    model.temperature.fill_(1.0 / scale)
    model.cal_bias.fill_(bias)

    # 2. Select the operating threshold on the (now recalibrated) dev segment.
    with torch.no_grad():
        dev_probability = model.calibrated_probability(dev_logits).numpy()
    threshold = threshold_at_fpr(dev.labels.numpy(), dev_probability, target_fpr)

    # 3. Real drift baseline from the dev segment's real event rows.
    model.feature_stats = feature_snapshot(dev)

    # 4. Verify on the untouched test segment.
    test_metrics = _evaluate(model, test)
    with torch.no_grad():
        test_probability = model.calibrated_probability(_logits(model, test)).numpy()
    confusion = _confusion_at(test.labels.numpy(), test_probability, threshold.tau)

    realized_fpr_ceiling = target_fpr * gates["max_realized_fpr_ratio"]
    gate_results = [
        _gate("dev_windows", dev_n, ">=", gates["min_windows_per_segment"]),
        _gate("test_windows", test_n, ">=", gates["min_windows_per_segment"]),
        _gate("dev_positives", dev_pos, ">=", gates["min_positives_per_segment"]),
        _gate("test_positives", test_pos, ">=", gates["min_positives_per_segment"]),
        _gate("roc_auc", test_metrics["roc_auc"], ">=", gates["min_roc_auc"]),
        _gate("ece", test_metrics["ece"], "<=", gates["max_ece"]),
        _gate("recall_at_target_fpr", test_metrics["recall_at_fpr_1pct"] if target_fpr == 0.01 else _recall_from_confusion(confusion), ">=", gates["min_recall_at_target_fpr"]),
        _gate("realized_fpr", confusion["realized_fpr"], "<=", realized_fpr_ceiling),
        _gate("feature_schema", 1.0 if model.feature_stats is not None else 0.0, ">=", 1.0),
    ]
    passed = all(gate["pass"] for gate in gate_results)
    binding = _is_binding(provenance)

    return {
        "report": "idr-intelligence-validation-v1",
        "engine_feature_schema_hash": feature_schema_hash(),
        "data_provenance": provenance,
        "binding": binding and passed,
        "verdict": "go" if passed else "no-go",
        "verdict_note": _verdict_note(passed, binding),
        "target_fpr": target_fpr,
        "operating_threshold": {
            "tau": None if threshold.tau == float("inf") else round(threshold.tau, 6),
            "dev_fpr": round(threshold.dev_fpr, 6),
            "dev_tpr": round(threshold.dev_tpr, 6),
        },
        "calibration": {"label": model.calibration_label(), "scale": round(scale, 6), "bias": round(bias, 6)},
        "data": {
            "windows": len(ordered),
            "dev": {"windows": dev_n, "positives": dev_pos},
            "test": {"windows": test_n, "positives": test_pos},
            "holdout": holdout,
        },
        "test_metrics": test_metrics,
        "test_confusion_at_threshold": confusion,
        "drift_baseline": {"sample_count": model.feature_stats["sample_count"], "features": len(model.feature_stats["histograms"])},
        "gates": gate_results,
    }


def _recall_from_confusion(confusion: dict[str, Any]) -> float:
    return confusion["realized_tpr"]


def _gate(name: str, value: float, op: str, threshold: float) -> dict[str, Any]:
    passed = value >= threshold if op == ">=" else value <= threshold
    return {"name": name, "value": round(float(value), 6), "op": op, "threshold": threshold, "pass": bool(passed)}


def _is_binding(provenance: str) -> bool:
    lowered = provenance.strip().lower()
    if not lowered:
        return False
    return not any(marker in lowered for marker in _SYNTHETIC_MARKERS)


def _verdict_note(passed: bool, binding: bool) -> str:
    if not passed:
        return "One or more gates failed; do not deploy. See the gates list."
    if not binding:
        return (
            "All gates pass, but the data provenance is unspecified or synthetic, so this is a "
            "WIRING verdict only — NOT a production sign-off. Re-run with real labeled campaigns and "
            "--data-provenance naming them to obtain a binding verdict."
        )
    return "All gates pass on operator-attested real data: cleared for shadow/advisory deployment."


def render_model_card(report: dict[str, Any]) -> str:
    """A model card markdown from a validation report — ships with the checkpoint."""
    metrics = report["test_metrics"]
    threshold = report["operating_threshold"]
    lines = [
        "# Model Card — idr-intelligence campaign scorer",
        "",
        f"**Verdict:** {report['verdict'].upper()}  ·  **Binding:** {report['binding']}",
        "",
        f"> {report['verdict_note']}",
        "",
        "## Intended use",
        "",
        "Advisory campaign-escalation scoring over `idr_common::IdrEvent` streams. Output is a",
        "hypothesis with evidence event IDs for the deterministic `idr-sentinel` correlator to",
        "corroborate. It must never trigger `PanicResponse` on its own.",
        "",
        "## Data & provenance",
        "",
        f"- Provenance (operator-attested): `{report['data_provenance']}`",
        f"- Windows: {report['data']['windows']} "
        f"(dev {report['data']['dev']['windows']}/{report['data']['dev']['positives']}+, "
        f"test {report['data']['test']['windows']}/{report['data']['test']['positives']}+, "
        f"holdout {report['data']['holdout']}, time-ordered)",
        f"- Feature schema hash: `{report['engine_feature_schema_hash']}`",
        f"- Drift baseline: {report['drift_baseline']['sample_count']} real event rows over "
        f"{report['drift_baseline']['features']} features",
        "",
        "## Operating point",
        "",
        f"- Target FPR: {report['target_fpr']}",
        f"- Threshold tau: {threshold['tau']}  (dev FPR {threshold['dev_fpr']}, dev TPR {threshold['dev_tpr']})",
        f"- Realized on test: FPR {report['test_confusion_at_threshold']['realized_fpr']}, "
        f"TPR {report['test_confusion_at_threshold']['realized_tpr']}, "
        f"precision {report['test_confusion_at_threshold']['precision']}",
        f"- Calibration: {report['calibration']['label']}",
        "",
        "## Held-out test metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| ROC-AUC | {metrics['roc_auc']} |",
        f"| PR-AUC | {metrics['pr_auc']} |",
        f"| Brier | {metrics['brier']} |",
        f"| ECE | {metrics['ece']} |",
        f"| recall@FPR=1% | {metrics['recall_at_fpr_1pct']} |",
        "",
        "## Gates",
        "",
        "| gate | value | requirement | pass |",
        "|---|---|---|---|",
        *[
            f"| {gate['name']} | {gate['value']} | {gate['op']} {gate['threshold']} | {'✅' if gate['pass'] else '❌'} |"
            for gate in report["gates"]
        ],
        "",
        "## Known limitations & non-goals",
        "",
        "- Not a standalone detector: advisory only, corroborated before any action.",
        "- Temporal-physics modes (`time_mode`, `decay_half_life`) ship off by default; see `reports/AUDIT.md`.",
        "- Trained/validated only on the attested data above; re-validate on schema or traffic changes.",
        "",
    ]
    return "\n".join(lines) + "\n"
