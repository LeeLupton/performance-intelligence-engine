from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_DIM, project_event
from .schema import IdrEvent


@dataclass(frozen=True)
class TemporalGraph:
    node_ids: tuple[str, ...]
    sequences: np.ndarray
    mask: np.ndarray
    adjacency: np.ndarray
    evidence_ids: tuple[tuple[str, ...], ...]
    relation_counts: dict[str, int]

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def build_temporal_graph(events: list[IdrEvent], max_steps: int = 24) -> TemporalGraph:
    if not events:
        raise ValueError("at least one event is required")
    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    projections = []
    previous = ordered[0].timestamp
    for event in ordered:
        delta = (event.timestamp - previous).total_seconds()
        projections.append(project_event(event, delta_seconds=delta))
        previous = event.timestamp

    node_ids = tuple(dict.fromkeys(entity for projection in projections for entity in projection.entities))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    histories = [[] for _ in node_ids]
    evidence = [[] for _ in node_ids]
    adjacency = np.eye(len(node_ids), dtype=np.float32)
    relation_counts: dict[str, int] = {}

    for projection in projections:
        for entity in projection.entities:
            idx = node_index[entity]
            histories[idx].append(projection.features)
            evidence[idx].append(projection.event.id)
        for left, right, relation in projection.edges:
            i, j = node_index[left], node_index[right]
            adjacency[i, j] = adjacency[j, i] = 1.0
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

    sequences = np.zeros((len(node_ids), max_steps, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(node_ids), max_steps), dtype=np.float32)
    for idx, history in enumerate(histories):
        clipped = history[-max_steps:]
        start = max_steps - len(clipped)
        if clipped:
            sequences[idx, start:] = np.stack(clipped)
            mask[idx, start:] = 1.0

    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degrees, 1.0), -0.5)
    adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return TemporalGraph(
        node_ids=node_ids,
        sequences=sequences,
        mask=mask,
        adjacency=adjacency,
        evidence_ids=tuple(tuple(dict.fromkeys(ids)) for ids in evidence),
        relation_counts=relation_counts,
    )
