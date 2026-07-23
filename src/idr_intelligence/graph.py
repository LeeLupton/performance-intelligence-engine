"""Temporal graph construction with per-node evidence provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .features import FEATURE_DIM, project_event
from .schema import IdrEvent

TIME_MODES = ("global", "per_entity", "time_aware")
_DELTA_FEATURE_INDEX = 2  # delta_seconds_log in FEATURE_NAMES


def _normalize_delta(seconds: float) -> float:
    """log1p seconds compressed to ~[0,1], matching project_event's feature scaling."""
    return float(np.log1p(max(seconds, 0.0)) / 12.0)


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

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def build_temporal_graph(events: list[IdrEvent], max_steps: int = 24, time_mode: str = "time_aware") -> TemporalGraph:
    """Order events, accumulate per-entity histories, and normalize the adjacency.

    time_mode controls the elapsed-time signal:
      - global:     delta_seconds_log stays the whole-stream inter-event gap (legacy)
      - per_entity: delta_seconds_log becomes the gap since THAT entity was last seen
      - time_aware: per-entity gap, and the deltas channel feeds S6 discretization
    The deltas array carries the per-entity gap for every mode; only the model's
    time-aware path consumes it.
    """
    if time_mode not in TIME_MODES:
        raise ValueError(f"unknown time_mode: {time_mode}")
    if not events:
        raise ValueError("at least one event is required")
    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    projections = []
    previous = ordered[0].timestamp
    for event in ordered:
        global_delta = (event.timestamp - previous).total_seconds()
        projections.append(project_event(event, delta_seconds=global_delta))
        previous = event.timestamp

    node_ids = tuple(dict.fromkeys(entity for projection in projections for entity in projection.entities))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    histories: list[list[np.ndarray]] = [[] for _ in node_ids]
    delta_histories: list[list[float]] = [[] for _ in node_ids]
    evidence: list[list[str]] = [[] for _ in node_ids]
    adjacency = np.eye(len(node_ids), dtype=np.float32)
    relation_counts: dict[str, int] = {}
    last_seen: dict[str, datetime] = {}

    for projection in projections:
        timestamp = projection.event.timestamp
        for entity in projection.entities:
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
            i, j = node_index[left], node_index[right]
            adjacency[i, j] = adjacency[j, i] = 1.0
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

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

    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degrees, 1.0), -0.5)
    adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return TemporalGraph(
        node_ids=node_ids,
        sequences=sequences,
        mask=mask,
        adjacency=adjacency,
        deltas=deltas,
        evidence_ids=tuple(tuple(dict.fromkeys(ids)) for ids in evidence),
        relation_counts=relation_counts,
    )
