"""Versioned benchmark suite: frozen manifests with regression floors.

A manifest pins the generator configuration and per-metric floors; the runner
retrains under that configuration and reports every floor violation. CI runs
the v1 manifest on every build, so a modeling change that degrades a scenario
past its floor fails the build instead of shipping silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .training import train_ablation


def run_benchmark(manifest_path: str | Path) -> dict[str, Any]:
    """Train under the manifest's generator config and check every floor."""
    manifest = json.loads(Path(manifest_path).read_text())
    report = train_ablation(**manifest["generator"])
    violations: list[str] = []
    for variant, floors in manifest["floors"]["variant_metrics"].items():
        row = report["metrics"][variant]
        if row["roc_auc"] < floors["roc_auc_min"]:
            violations.append(f"{variant}: roc_auc {row['roc_auc']} < floor {floors['roc_auc_min']}")
        if row["brier"] > floors["brier_max"]:
            violations.append(f"{variant}: brier {row['brier']} > ceiling {floors['brier_max']}")
    for scenario, brier_max in manifest["floors"]["scenario_brier_max"].items():
        brier = report["scenario_generalization"][scenario]["brier"]
        if brier > brier_max:
            violations.append(f"scenario {scenario}: brier {brier} > ceiling {brier_max}")
    return {
        "suite_version": manifest["suite_version"],
        "passed": not violations,
        "violations": violations,
        "report": report,
    }
