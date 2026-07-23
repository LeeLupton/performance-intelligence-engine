"""Single typed home for every tunable engine constant.

The prior tables live here as the canonical defaults; schema.py re-exports
them so feature extraction keeps one import path. configs/default.toml must
reproduce DEFAULT_CONFIG exactly — a test enforces the no-op guarantee.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENGINE_VERSION = "0.1.0"

DEFAULT_SEVERITY_WEIGHT = {
    "INFO": 0.05,
    "WARNING": 0.25,
    "HIGH": 0.65,
    "CRITICAL": 0.85,
    "IMPOSSIBLE": 1.0,
}

DEFAULT_KIND_PRIOR = {
    "socket_lineage": 0.12,
    "suspicious_beacon": 0.72,
    "physics_anomaly": 0.68,
    "octet_reversal_detected": 0.74,
    "ntp_time_shift": 0.42,
    "hsts_time_manipulation": 0.78,
    "nvme_latency_anomaly": 0.52,
    "mac_flapping": 0.64,
    "rtc_clock_divergence": 0.38,
    "triage_classification": 0.60,
    "bgp_anomaly": 0.48,
    "impossible_state": 1.0,
}


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions and pooling mode used for every trained variant."""

    hidden_dim: int = 24
    state_dim: int = 6
    pooling: str = "attention"


@dataclass(frozen=True)
class GraphConfig:
    """History-window and padding budgets; scoring keeps a longer window than training."""

    score_max_steps: int = 24
    train_max_steps: int = 16
    train_max_nodes: int = 24


@dataclass(frozen=True)
class ScoringConfig:
    """Finding-shaping knobs."""

    top_k: int = 5


@dataclass(frozen=True)
class EngineConfig:
    """The complete tunable surface of the engine."""

    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        """Deterministic identity of this configuration for manifests."""
        return hashlib.blake2s(json.dumps(self.to_dict(), sort_keys=True).encode(), digest_size=16).hexdigest()


DEFAULT_CONFIG = EngineConfig()


def load_config(path: str | Path) -> EngineConfig:
    """Load an EngineConfig from TOML; absent sections keep their defaults."""
    data = tomllib.loads(Path(path).read_text())
    return EngineConfig(
        model=ModelConfig(**data.get("model", {})),
        graph=GraphConfig(**data.get("graph", {})),
        scoring=ScoringConfig(**data.get("scoring", {})),
    )
