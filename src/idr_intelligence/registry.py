"""Model manifests: feature-schema identity and provenance for checkpoints.

The feature schema hash covers FEATURE_NAMES plus the prior tables actually
used at extraction time, so any change to feature semantics changes the hash
and a stale checkpoint refuses to load instead of silently mis-scoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from .config import DEFAULT_CONFIG, ENGINE_VERSION, EngineConfig
from .features import FEATURE_NAMES
from .schema import KIND_PRIOR, SEVERITY_WEIGHT


class SchemaMismatchError(RuntimeError):
    """Checkpoint was trained under a different feature schema than this runtime."""


def feature_schema_hash() -> str:
    """Identity of the runtime feature semantics: names plus prior tables."""
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "severity_weight": SEVERITY_WEIGHT,
        "kind_prior": KIND_PRIOR,
    }
    return hashlib.blake2s(json.dumps(payload, sort_keys=True).encode(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class ModelManifest:
    """Provenance embedded in every checkpoint and mirrored into findings."""

    engine_version: str
    feature_schema_hash: str
    config_hash: str
    created_at: str
    calibration: str = "none"

    @classmethod
    def create(cls, config: EngineConfig = DEFAULT_CONFIG, calibration: str = "none") -> ModelManifest:
        return cls(
            engine_version=ENGINE_VERSION,
            feature_schema_hash=feature_schema_hash(),
            config_hash=config.config_hash(),
            created_at=datetime.now(UTC).isoformat(),
            calibration=calibration,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelManifest:
        return cls(**raw)

    def verify_feature_schema(self) -> None:
        """Raise SchemaMismatchError if this manifest disagrees with the runtime."""
        current = feature_schema_hash()
        if self.feature_schema_hash != current:
            raise SchemaMismatchError(
                f"checkpoint feature schema {self.feature_schema_hash} does not match "
                f"runtime feature schema {current}; retrain or pin the matching engine version"
            )
