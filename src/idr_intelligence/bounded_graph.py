"""Bounded entity-graph memory: a node budget with an audited LRU eviction trail.

At demo scale (graphs of a few dozen nodes) the budget never binds; it exists so
a streaming scorer (W15) over an unbounded real event stream keeps a fixed memory
footprint, and so every dropped entity leaves an auditable record rather than
silently vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EvictionRecord:
    """One entity dropped to stay within the node budget."""

    entity: str
    last_seen: datetime
    reason: str


@dataclass(frozen=True)
class GraphBudget:
    """Keep at most max_nodes entities, evicting the least-recently-seen first."""

    max_nodes: int

    def apply(self, last_seen: dict[str, datetime]) -> tuple[tuple[str, ...], tuple[EvictionRecord, ...]]:
        """Return the kept entities (input order) and eviction records for the rest.

        Ties on last_seen are broken by entity id so the result is deterministic.
        """
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if len(last_seen) <= self.max_nodes:
            return tuple(last_seen), ()
        ranked = sorted(last_seen, key=lambda entity: (last_seen[entity], entity), reverse=True)
        kept = set(ranked[: self.max_nodes])
        evictions = tuple(
            EvictionRecord(entity=entity, last_seen=last_seen[entity], reason="node_budget")
            for entity in ranked[self.max_nodes:]
        )
        return tuple(entity for entity in last_seen if entity in kept), evictions
