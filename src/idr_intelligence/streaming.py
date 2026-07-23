"""One-event-at-a-time campaign scoring over carried S6 state.

The batch pipeline replays every entity's full history on each call. The
streaming scorer instead carries each entity's S6 (state, output) forward with
the W13 step() cell — O(1) work and O(1) state per (event, entity) — and runs
the model's relational head (GNN + pooling + heads, the exact module forward()
uses) over the carried states only when a finding is requested. Entity memory
is bounded by GraphBudget with an audited eviction trail.

What streaming deliberately does NOT provide: occlusion-based entity_evidence
(it requires replaying event histories, which is exactly the memory streaming
exists to avoid) — findings carry the trained attention ranking, evidence IDs,
stages, and drift, with entity_evidence empty. Events must arrive in
chronological order; an upstream replay/sort buffer owns reordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from .attack import KIND_TO_ATTACK, next_stage_from_stages
from .bounded_graph import EvictionRecord, GraphBudget
from .campaigns import CampaignRegistry
from .config import DEFAULT_CONFIG, ENGINE_VERSION
from .evidence import apply_suppressions
from .features import project_event
from .graph import _DELTA_FEATURE_INDEX, _normalize_delta, degree_normalize
from .models import CampaignModel
from .pipeline import IntelligenceFinding, psi_drift
from .registry import feature_schema_hash
from .schema import IdrEvent

EVIDENCE_LIMIT = 64  # most-recent evidence event ids kept per entity


@dataclass
class EntityState:
    """One entity's carried scan state — everything streaming keeps per node."""

    state: torch.Tensor  # [1, hidden, state_dim]
    output: torch.Tensor  # [1, hidden]
    last_seen: datetime
    event_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)


class StreamingScorer:
    """Incremental scorer: ingest() events one by one, ask for a finding() anytime.

    Requires an S6 model (the production hybrid): the static-baseline ablation
    pools whole histories and has no carried state to stream. The relational
    head runs over N carried states at query time — cheap and skew-free, since
    it is the same module forward() uses.
    """

    def __init__(
        self,
        model: CampaignModel,
        budget: GraphBudget | None = None,
        model_version: str = "development",
    ) -> None:
        if model.temporal is None:
            raise ValueError("StreamingScorer requires an S6 model (use_s6=True); the static baseline has no carried state to stream")
        model.eval()
        self.model = model
        self.budget = budget
        self.model_version = model_version
        self.entities: dict[str, EntityState] = {}
        self.evictions: list[EvictionRecord] = []
        self.relation_counts: dict[str, int] = {}
        self._typed_edges: dict[tuple[str, str, str], None] = {}
        self._edge_last_seen: dict[tuple[str, str], datetime] = {}
        self._stages: dict[str, dict[str, Any]] = {}
        self._previous_time: datetime | None = None
        self._first_event: tuple[datetime, str] | None = None
        self._drift_counts: np.ndarray | None = None
        if model.feature_stats:
            bins = len(model.feature_stats["bin_edges"]) - 1
            self._drift_counts = np.zeros((model.feature_dim, bins), dtype=np.int64)
        self.events_seen = 0

    def ingest(self, event: IdrEvent) -> None:
        """Advance every entity the event mentions by one S6 step; update the graph."""
        if self._previous_time is not None and event.timestamp < self._previous_time:
            raise ValueError(
                f"out-of-order event {event.id}: {event.timestamp.isoformat()} precedes the stream clock "
                f"{self._previous_time.isoformat()}; sort or buffer upstream"
            )
        global_delta = (event.timestamp - self._previous_time).total_seconds() if self._previous_time is not None else 0.0
        projection = project_event(event, delta_seconds=global_delta)
        self._previous_time = event.timestamp
        if self._first_event is None or (event.timestamp, event.id) < self._first_event:
            self._first_event = (event.timestamp, event.id)
        mapping = KIND_TO_ATTACK.get(event.kind_type)
        if mapping is not None and event.kind_type not in self._stages:
            self._stages[event.kind_type] = {
                "tactic": mapping["tactic"],
                "technique": mapping["technique"],
                "kind_type": event.kind_type,
                "first_event_id": event.id,
            }
        assert self.model.temporal is not None
        with torch.no_grad():
            for entity in projection.entities:
                existing = self.entities.get(entity)
                gap_seconds = (event.timestamp - existing.last_seen).total_seconds() if existing is not None else 0.0
                entity_delta = _normalize_delta(gap_seconds)
                if self.model.time_mode == "global":
                    features = projection.features
                else:
                    features = projection.features.copy()
                    features[_DELTA_FEATURE_INDEX] = entity_delta
                if existing is None:
                    state, output = self.model.temporal.initial_state(1, torch.device("cpu"), torch.float32)
                    existing = EntityState(state=state, output=output, last_seen=event.timestamp)
                    self.entities[entity] = existing
                x_t = torch.from_numpy(features).unsqueeze(0)
                delta_t = torch.tensor([entity_delta], dtype=torch.float32) if self.model.time_mode == "time_aware" else None
                existing.state, existing.output = self.model.temporal.step(x_t, existing.state, existing.output, delta_t=delta_t)
                existing.last_seen = event.timestamp
                existing.event_count += 1
                if event.id not in existing.evidence_ids:
                    existing.evidence_ids.append(event.id)
                    del existing.evidence_ids[:-EVIDENCE_LIMIT]
                if self._drift_counts is not None:
                    assert self.model.feature_stats is not None
                    edges = np.asarray(self.model.feature_stats["bin_edges"])
                    for index in range(features.shape[0]):
                        self._drift_counts[index] += np.histogram(features[index : index + 1], bins=edges)[0]
        for left, right, relation in projection.edges:
            self._edge_last_seen[(min(left, right), max(left, right))] = event.timestamp
            self.relation_counts[relation] = self.relation_counts.get(relation, 0) + 1
            self._typed_edges[(left, right, relation)] = None
        self.events_seen += 1
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        """Evict least-recently-seen entities past the budget, with an audit trail."""
        if self.budget is None or len(self.entities) <= self.budget.max_nodes:
            return
        kept, evicted = self.budget.apply({entity: state.last_seen for entity, state in self.entities.items()})
        keep = set(kept)
        self.evictions.extend(evicted)
        for record in evicted:
            del self.entities[record.entity]
        self._edge_last_seen = {pair: seen for pair, seen in self._edge_last_seen.items() if pair[0] in keep and pair[1] in keep}
        self._typed_edges = {edge: None for edge in self._typed_edges if edge[0] in keep and edge[1] in keep}

    def finding(
        self,
        top_k: int = DEFAULT_CONFIG.scoring.top_k,
        suppressions: list[str] | None = None,
        registry: CampaignRegistry | None = None,
    ) -> IntelligenceFinding:
        """Score the carried state now and return an IntelligenceFinding.

        entity_evidence is empty by design (occlusion needs history replay);
        everything else — calibrated probability, trained attention ranking,
        evidence IDs, ATT&CK stages, drift, campaign identity — is carried.
        """
        if not self.entities:
            raise ValueError("no events ingested")
        assert self.model.temporal is not None
        node_ids = tuple(self.entities)
        index_of = {entity: index for index, entity in enumerate(node_ids)}
        adjacency = np.eye(len(node_ids), dtype=np.float32)
        assert self._previous_time is not None
        for (left, right), seen in self._edge_last_seen.items():
            weight = np.float32(1.0)
            if self.model.decay_half_life is not None:
                age = (self._previous_time - seen).total_seconds()
                weight = np.float32(0.5 ** (age / self.model.decay_half_life))
            i, j = index_of[left], index_of[right]
            adjacency[i, j] = adjacency[j, i] = weight
        adjacency = degree_normalize(adjacency)
        with torch.no_grad():
            outputs = torch.cat([state.output for state in self.entities.values()], dim=0)
            node_state = self.model.temporal.norm(outputs).unsqueeze(0)
            active = torch.ones(1, len(node_ids), 1, dtype=node_state.dtype)
            output = self.model.relational_head(node_state, active, torch.from_numpy(adjacency).unsqueeze(0))
            raw_probability = torch.sigmoid(output.graph_logit)[0].item()
            probability = self.model.calibrated_probability(output.graph_logit)[0].item()
            node_probability = torch.sigmoid(output.node_logits[0]).cpu().numpy()
        ranking_scores, applied_suppressions = apply_suppressions(node_ids, node_probability, suppressions or [])
        ranked = np.argsort(-ranking_scores)[: min(top_k, len(node_ids))]
        ranked = np.array([index for index in ranked if np.isfinite(ranking_scores[index])], dtype=int)
        related = tuple(node_ids[index] for index in ranked)
        evidence = tuple(dict.fromkeys(event_id for index in ranked for event_id in self.entities[node_ids[index]].evidence_ids))
        stages = tuple(self._stages.values())
        scored_at = datetime.now(UTC).isoformat()
        if registry is not None:
            campaign_id, continues_campaign, windows_observed = registry.match_or_register(node_ids, scored_at)
        else:
            assert self._first_event is not None
            campaign_id, continues_campaign, windows_observed = f"idr-campaign-{self._first_event[1][:8]}", False, 1
        drift = None
        if self._drift_counts is not None:
            assert self.model.feature_stats is not None
            drift = psi_drift(self.model.feature_stats, self._drift_counts)
        return IntelligenceFinding(
            campaign_id=campaign_id,
            escalation_probability=round(probability, 6),
            raw_escalation_probability=round(raw_probability, 6),
            calibration=self.model.calibration_label(),
            predicted_next_stage=next_stage_from_stages(stages),
            observed_attack_stages=stages,
            related_entities=related,
            entity_evidence=(),
            applied_suppressions=applied_suppressions,
            evidence_event_ids=evidence,
            model_version=self.model_version,
            graph_nodes=len(node_ids),
            graph_relations=dict(self.relation_counts),
            engine_version=ENGINE_VERSION,
            feature_schema_hash=feature_schema_hash(),
            scored_at=scored_at,
            feature_drift=drift,
            continues_campaign=continues_campaign,
            windows_observed=windows_observed,
        )
