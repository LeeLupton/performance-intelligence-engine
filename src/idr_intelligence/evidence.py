"""Structured per-entity evidence: why the model flagged each ranked entity.

Turns the flat top-k entity list into an analyst-facing record — for each
ranked entity, the events that implicate it, the typed edges tying it to other
flagged entities, the features that most drove its score (occlusion
attribution), and the ATT&CK techniques its events touched. All of it is
derived from the same graph and model the finding already uses; nothing here
is a new model output that could bypass the evidence boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .attack import KIND_TO_ATTACK
from .features import FEATURE_NAMES
from .graph import TemporalGraph
from .models import CampaignModel
from .schema import IdrEvent


@dataclass(frozen=True)
class EntityEvidence:
    """Why one ranked entity is on the list."""

    entity: str
    node_probability: float
    evidence_event_ids: tuple[str, ...]
    related_edges: tuple[dict[str, str], ...]
    top_features: tuple[dict[str, Any], ...]
    attack_techniques: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def occlusion_attribution(
    model: CampaignModel,
    sequence: torch.Tensor,
    mask: torch.Tensor,
    adjacency: torch.Tensor,
    deltas: torch.Tensor,
    baseline_node_logits: torch.Tensor,
) -> np.ndarray:
    """Per-node, per-feature score drop when each feature channel is zeroed.

    attribution[node, feature] = baseline_node_logit − logit with that feature
    zeroed across the whole graph; positive means the feature pushed the node up.
    One forward pass per feature (cheap for a single scoring call).
    """
    feature_dim = sequence.shape[-1]
    nodes = sequence.shape[1]
    attribution = np.zeros((nodes, feature_dim), dtype=np.float32)
    with torch.no_grad():
        for channel in range(feature_dim):
            occluded = sequence.clone()
            occluded[..., channel] = 0.0
            logits = model(occluded, mask, adjacency, deltas).node_logits[0]
            attribution[:, channel] = (baseline_node_logits - logits).cpu().numpy()
    return attribution


def build_entity_evidence(
    graph: TemporalGraph,
    events: list[IdrEvent],
    node_probability: np.ndarray,
    ranked: np.ndarray,
    attribution: np.ndarray,
    top_features: int = 3,
) -> tuple[EntityEvidence, ...]:
    """Assemble one EntityEvidence per ranked entity from graph + model artifacts."""
    ranked_ids = {graph.node_ids[index] for index in ranked}
    events_by_id = {event.id: event for event in events}
    records = []
    for index in ranked:
        entity = graph.node_ids[index]
        related = tuple(
            {"peer": right if left == entity else left, "relation": relation}
            for left, right, relation in graph.typed_edges
            if (left == entity and right in ranked_ids) or (right == entity and left in ranked_ids)
        )
        order = np.argsort(-attribution[index])[:top_features]
        feature_records = tuple(
            {"feature": FEATURE_NAMES[channel], "attribution": round(float(attribution[index][channel]), 6)}
            for channel in order
            if attribution[index][channel] > 0
        )
        techniques = []
        for event_id in graph.evidence_ids[index]:
            event = events_by_id.get(event_id)
            mapping = KIND_TO_ATTACK.get(event.kind_type) if event else None
            if mapping:
                techniques.append(mapping["technique"])
        records.append(EntityEvidence(
            entity=entity,
            node_probability=round(float(node_probability[index]), 6),
            evidence_event_ids=graph.evidence_ids[index],
            related_edges=related,
            top_features=feature_records,
            attack_techniques=tuple(dict.fromkeys(techniques)),
        ))
    return tuple(records)


def apply_suppressions(node_ids: tuple[str, ...], node_probability: np.ndarray, suppressions: list[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Attenuate suppressed entities out of the ranking; never hide the finding.

    A suppression is an exact entity id or a prefix ending in ':' (e.g.
    'ip:203.0.113.5' or 'host:'). Matches are pushed below the ranking floor so
    they drop out of the top-k, but they remain in the graph and the model's
    campaign probability is untouched — idr-sentinel still receives the finding.
    Returns the adjusted probabilities and the list of entities actually matched.
    """
    if not suppressions:
        return node_probability, ()
    adjusted = node_probability.copy()
    matched = []
    for index, entity in enumerate(node_ids):
        if any(entity == rule or (rule.endswith(":") and entity.startswith(rule)) for rule in suppressions):
            adjusted[index] = -np.inf
            matched.append(entity)
    return adjusted, tuple(matched)
