"""Evidence-linked inference over IdrEvent streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from .attack import observed_attack_stages, predict_next_stage
from .campaigns import CampaignRegistry
from .config import DEFAULT_CONFIG, ENGINE_VERSION
from .evidence import apply_suppressions, build_entity_evidence, occlusion_attribution
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
    entity_evidence: tuple[dict[str, Any], ...]
    applied_suppressions: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    model_version: str
    graph_nodes: int
    graph_relations: dict[str, int]
    engine_version: str
    feature_schema_hash: str
    scored_at: str
    feature_drift: dict[str, Any] | None
    continues_campaign: bool = False
    windows_observed: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def score_events(
    events: list[IdrEvent],
    model: CampaignModel,
    model_version: str = "development",
    max_steps: int = DEFAULT_CONFIG.graph.score_max_steps,
    top_k: int = DEFAULT_CONFIG.scoring.top_k,
    suppressions: list[str] | None = None,
    registry: CampaignRegistry | None = None,
) -> IntelligenceFinding:
    """Score one event set and return a finding with ranked entities and evidence.

    suppressions (per-deployment allowlist) attenuate matching entities out of
    the ranking without hiding the finding or touching the campaign probability.
    registry, when provided, resolves a stable campaign identity across scoring
    windows by durable-entity fingerprint matching (the registry is mutated:
    matched campaigns are extended, unmatched ones registered). Without it the
    campaign id is derived from the window's first event, unique per call.
    """
    graph = build_temporal_graph(events, max_steps=max_steps, time_mode=model.time_mode, decay_half_life=model.decay_half_life)
    first = min(events, key=lambda event: (event.timestamp, event.id))
    sequence = torch.from_numpy(graph.sequences).unsqueeze(0)
    mask = torch.from_numpy(graph.mask).unsqueeze(0)
    adjacency = torch.from_numpy(graph.adjacency).unsqueeze(0)
    deltas = torch.from_numpy(graph.deltas).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(sequence, mask, adjacency, deltas)
        raw_probability = torch.sigmoid(output.graph_logit)[0].item()
        probability = model.calibrated_probability(output.graph_logit)[0].item()
        node_logits = output.node_logits[0]
        node_probability = torch.sigmoid(node_logits).cpu().numpy()
    calibration = model.calibration_label()
    ranking_scores, applied_suppressions = apply_suppressions(graph.node_ids, node_probability, suppressions or [])
    ranked = np.argsort(-ranking_scores)[: min(top_k, graph.node_count)]
    ranked = np.array([index for index in ranked if np.isfinite(ranking_scores[index])], dtype=int)
    related = tuple(graph.node_ids[index] for index in ranked)
    evidence = tuple(dict.fromkeys(event_id for index in ranked for event_id in graph.evidence_ids[index]))
    attribution = occlusion_attribution(model, sequence, mask, adjacency, deltas, node_logits)
    entity_evidence = tuple(record.to_dict() for record in build_entity_evidence(graph, events, node_probability, ranked, attribution))
    scored_at = datetime.now(UTC).isoformat()
    if registry is not None:
        campaign_id, continues_campaign, windows_observed = registry.match_or_register(graph.node_ids, scored_at)
    else:
        campaign_id, continues_campaign, windows_observed = f"idr-campaign-{first.id[:8]}", False, 1
    return IntelligenceFinding(
        campaign_id=campaign_id,
        escalation_probability=round(probability, 6),
        raw_escalation_probability=round(raw_probability, 6),
        calibration=calibration,
        predicted_next_stage=predict_next_stage(events),
        observed_attack_stages=observed_attack_stages(events),
        related_entities=related,
        entity_evidence=entity_evidence,
        applied_suppressions=applied_suppressions,
        evidence_event_ids=evidence,
        model_version=model_version,
        graph_nodes=graph.node_count,
        graph_relations=graph.relation_counts,
        engine_version=ENGINE_VERSION,
        feature_schema_hash=feature_schema_hash(),
        scored_at=scored_at,
        feature_drift=_feature_drift(model, graph),
        continues_campaign=continues_campaign,
        windows_observed=windows_observed,
    )


def psi_drift(stats: dict[str, Any], observed_counts: np.ndarray, flag_threshold: float = 0.2) -> dict[str, Any]:
    """PSI of per-feature observed histogram counts against a training snapshot.

    observed_counts is [features, bins], binned on stats["bin_edges"]. Shared by
    batch scoring (which histograms the window's rows) and the streaming scorer
    (which accumulates counts one event at a time). PSI >= 0.2 per feature is
    the conventional 'investigate' line.
    """
    psi_values = []
    for index, train_counts in enumerate(stats["histograms"]):
        train_share = np.asarray(train_counts, dtype=np.float64) + 1e-4
        observed_share = observed_counts[index].astype(np.float64) + 1e-4
        train_share /= train_share.sum()
        observed_share /= observed_share.sum()
        psi_values.append(float(((observed_share - train_share) * np.log(observed_share / train_share)).sum()))
    flagged = [FEATURE_NAMES[index] for index, value in enumerate(psi_values) if value >= flag_threshold]
    return {
        "psi_max": round(max(psi_values), 6),
        "psi_mean": round(float(np.mean(psi_values)), 6),
        "flagged_features": flagged,
    }


def _feature_drift(model: CampaignModel, graph: TemporalGraph) -> dict[str, Any] | None:
    """Advisory PSI of scored event features against the training snapshot.

    Returns None when the model carries no snapshot (hand-built or legacy models).
    """
    stats = model.feature_stats
    if not stats:
        return None
    rows = graph.sequences[graph.mask > 0]
    edges = np.asarray(stats["bin_edges"])
    observed = np.stack([np.histogram(rows[:, index], bins=edges)[0] for index in range(rows.shape[1])])
    return psi_drift(stats, observed)


