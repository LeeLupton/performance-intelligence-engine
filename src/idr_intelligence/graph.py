"""Temporal graph construction with per-node evidence provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .bounded_graph import EvictionRecord, GraphBudget
from .features import FEATURE_DIM, project_event
from .schema import IdrEvent

TIME_MODES = ("global", "per_entity", "time_aware")
_DELTA_FEATURE_INDEX = 2  # delta_seconds_log in FEATURE_NAMES


def _normalize_delta(seconds: float) -> float:
    """log1p seconds compressed to ~[0,1], matching project_event's feature scaling."""
    return float(np.log1p(max(seconds, 0.0)) / 12.0)


def degree_normalize(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric D^-1/2 A D^-1/2 normalization — shared by batch and streaming paths."""
    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degrees, 1.0), -0.5)
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


@dataclass(frozen=True)
class TemporalGraph:
    """Entity nodes with ordered feature histories, per-entity time deltas, adjacency, evidence."""

    node_ids: tuple[str, ...]
    sequences: np.ndarray
    mask: np.ndarray
    adjacency: np.ndarray
    deltas: np.ndarray
    evidence_ids: tuple[tuple[str, ...], ...]
    relation_counts: dict[str, int]
    typed_edges: tuple[tuple[str, str, str], ...] = ()
    evictions: tuple[EvictionRecord, ...] = ()

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def build_temporal_graph(
    events: list[IdrEvent],
    max_steps: int = 24,
    time_mode: str = "global",
    decay_half_life: float | None = None,
    budget: GraphBudget | None = None,
) -> TemporalGraph:
    """Order events, accumulate per-entity histories, and normalize the adjacency.

    time_mode controls the elapsed-time signal:
      - global:     delta_seconds_log stays the whole-stream inter-event gap (legacy)
      - per_entity: delta_seconds_log becomes the gap since THAT entity was last seen
      - time_aware: per-entity gap, and the deltas channel feeds S6 discretization
    decay_half_life (seconds): with a value, off-diagonal edges are weighted by
    0.5 ** (age / half_life) where age is the time since the edge was last
    reinforced, so stale relationships fade. None keeps binary edges (legacy).
    budget bounds the node set, evicting least-recently-seen entities with an
    audit trail; None is unbounded (the demo default).
    """
    if time_mode not in TIME_MODES:
        raise ValueError(f"unknown time_mode: {time_mode}")
    if decay_half_life is not None and decay_half_life <= 0:
        raise ValueError("decay_half_life must be positive or None")
    if not events:
        raise ValueError("at least one event is required")
    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    last_event_time = ordered[-1].timestamp
    projections = []
    previous = ordered[0].timestamp
    for event in ordered:
        global_delta = (event.timestamp - previous).total_seconds()
        projections.append(project_event(event, delta_seconds=global_delta))
        previous = event.timestamp

    # Pass 1: entity order + last-seen, to apply the node budget before building.
    entity_last_seen: dict[str, datetime] = {}
    for projection in projections:
        for entity in projection.entities:
            entity_last_seen[entity] = projection.event.timestamp
    if budget is not None:
        node_ids, evictions = budget.apply(entity_last_seen)
    else:
        node_ids, evictions = tuple(entity_last_seen), ()
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}

    histories: list[list[np.ndarray]] = [[] for _ in node_ids]
    delta_histories: list[list[float]] = [[] for _ in node_ids]
    evidence: list[list[str]] = [[] for _ in node_ids]
    adjacency = np.eye(len(node_ids), dtype=np.float32)
    edge_last_seen: dict[tuple[int, int], datetime] = {}
    relation_counts: dict[str, int] = {}
    typed_edges: dict[tuple[str, str, str], None] = {}
    last_seen: dict[str, datetime] = {}

    for projection in projections:
        timestamp = projection.event.timestamp
        for entity in projection.entities:
            if entity not in node_index:
                continue
            idx = node_index[entity]
            gap_seconds = (timestamp - last_seen[entity]).total_seconds() if entity in last_seen else 0.0
            last_seen[entity] = timestamp
            entity_delta = _normalize_delta(gap_seconds)
            if time_mode == "global":
                features = projection.features
            else:
                # Per-entity delta replaces the global gap in the feature row.
                features = projection.features.copy()
                features[_DELTA_FEATURE_INDEX] = entity_delta
            histories[idx].append(features)
            delta_histories[idx].append(entity_delta)
            evidence[idx].append(projection.event.id)
        for left, right, relation in projection.edges:
            if left not in node_index or right not in node_index:
                continue
            i, j = node_index[left], node_index[right]
            adjacency[i, j] = adjacency[j, i] = 1.0
            edge_last_seen[(min(i, j), max(i, j))] = timestamp
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            typed_edges[(left, right, relation)] = None

    if decay_half_life is not None:
        for (i, j), seen in edge_last_seen.items():
            age = (last_event_time - seen).total_seconds()
            weight = np.float32(0.5 ** (age / decay_half_life))
            adjacency[i, j] = adjacency[j, i] = weight

    sequences = np.zeros((len(node_ids), max_steps, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(node_ids), max_steps), dtype=np.float32)
    deltas = np.zeros((len(node_ids), max_steps), dtype=np.float32)
    for idx, history in enumerate(histories):
        clipped = history[-max_steps:]
        clipped_deltas = delta_histories[idx][-max_steps:]
        start = max_steps - len(clipped)
        if clipped:
            sequences[idx, start:] = np.stack(clipped)
            mask[idx, start:] = 1.0
            deltas[idx, start:] = np.asarray(clipped_deltas, dtype=np.float32)

    adjacency = degree_normalize(adjacency)
    return TemporalGraph(
        node_ids=node_ids,
        sequences=sequences,
        mask=mask,
        adjacency=adjacency,
        deltas=deltas,
        evidence_ids=tuple(tuple(dict.fromkeys(ids)) for ids in evidence),
        relation_counts=relation_counts,
        typed_edges=tuple(typed_edges),
        evictions=evictions,
    )
