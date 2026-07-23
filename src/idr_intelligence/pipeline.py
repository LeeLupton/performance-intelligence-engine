"""Evidence-linked inference over IdrEvent streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import torch

from typing import Any

from .attack import observed_attack_stages, predict_next_stage
from .config import DEFAULT_CONFIG, ENGINE_VERSION
from .features import FEATURE_NAMES
from .graph import TemporalGraph, build_temporal_graph
from .models import CampaignModel
from .registry import feature_schema_hash
from .schema import IdrEvent


@dataclass(frozen=True)
class IntelligenceFinding:
    """Advisory campaign hypothesis carrying the event IDs that back it."""

    campaign_id: str
    escalation_probability: float
    raw_escalation_probability: float
    calibration: str
    predicted_next_stage: str
    observed_attack_stages: tuple[dict[str, Any], ...]
    related_entities: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    model_version: str
    graph_nodes: int
    graph_relations: dict[str, int]
    engine_version: str
    feature_schema_hash: str
    scored_at: str
    feature_drift: dict[str, Any] | None

    def to_dict(self) -> dict:
        return asdict(self)


def score_events(
    events: list[IdrEvent],
    model: CampaignModel,
    model_version: str = "development",
    max_steps: int = DEFAULT_CONFIG.graph.score_max_steps,
    top_k: int = DEFAULT_CONFIG.scoring.top_k,
) -> IntelligenceFinding:
    """Score one event set and return a finding with ranked entities and evidence."""
    graph = build_temporal_graph(events, max_steps=max_steps)
    first = min(events, key=lambda event: (event.timestamp, event.id))
    sequence = torch.from_numpy(graph.sequences).unsqueeze(0)
    mask = torch.from_numpy(graph.mask).unsqueeze(0)
    adjacency = torch.from_numpy(graph.adjacency).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(sequence, mask, adjacency)
        raw_probability = torch.sigmoid(output.graph_logit)[0].item()
        probability = model.calibrated_probability(output.graph_logit)[0].item()
        node_probability = torch.sigmoid(output.node_logits)[0].cpu().numpy()
    temperature = float(model.temperature.item())
    calibration = "none" if temperature == 1.0 else f"temperature:{temperature:.6f}"
    ranked = np.argsort(-node_probability)[: min(top_k, graph.node_count)]
    related = tuple(graph.node_ids[index] for index in ranked)
    evidence = tuple(dict.fromkeys(event_id for index in ranked for event_id in graph.evidence_ids[index]))
    return IntelligenceFinding(
        campaign_id=f"idr-campaign-{first.id[:8]}",
        escalation_probability=round(probability, 6),
        raw_escalation_probability=round(raw_probability, 6),
        calibration=calibration,
        predicted_next_stage=predict_next_stage(events),
        observed_attack_stages=observed_attack_stages(events),
        related_entities=related,
        evidence_event_ids=evidence,
        model_version=model_version,
        graph_nodes=graph.node_count,
        graph_relations=graph.relation_counts,
        engine_version=ENGINE_VERSION,
        feature_schema_hash=feature_schema_hash(),
        scored_at=datetime.now(timezone.utc).isoformat(),
        feature_drift=_feature_drift(model, graph),
    )


def _feature_drift(model: CampaignModel, graph: TemporalGraph, flag_threshold: float = 0.2) -> dict[str, Any] | None:
    """Advisory PSI of scored event features against the training snapshot.

    Returns None when the model carries no snapshot (hand-built or legacy
    models). PSI >= 0.2 per feature is the conventional 'investigate' line.
    """
    stats = model.feature_stats
    if not stats:
        return None
    rows = graph.sequences[graph.mask > 0]
    edges = np.asarray(stats["bin_edges"])
    psi_values = []
    for index, train_counts in enumerate(stats["histograms"]):
        observed_counts = np.histogram(rows[:, index], bins=edges)[0]
        train_share = np.asarray(train_counts, dtype=np.float64) + 1e-4
        observed_share = observed_counts.astype(np.float64) + 1e-4
        train_share /= train_share.sum()
        observed_share /= observed_share.sum()
        psi_values.append(float(((observed_share - train_share) * np.log(observed_share / train_share)).sum()))
    flagged = [FEATURE_NAMES[index] for index, value in enumerate(psi_values) if value >= flag_threshold]
    return {
        "psi_max": round(max(psi_values), 6),
        "psi_mean": round(float(np.mean(psi_values)), 6),
        "flagged_features": flagged,
    }


