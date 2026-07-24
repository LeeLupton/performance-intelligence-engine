"""ONNX export of the streaming inference surface, and its reference runner.

The exported bundle is the cross-language serving contract (W17): the W13/W15
step() cell and the shared relational head become two ONNX graphs, and every
constant the Rust bridge needs to reproduce scoring — prior tables, ATT&CK
mapping, calibration, dimensions, IO signatures — travels in manifest.json so
nothing is hardcoded twice. OnnxStreamScorer drives those graphs exactly the
way StreamingScorer drives the torch modules; it is the executable spec the
Rust port is tested against.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .attack import KIND_TO_ATTACK, TACTIC_ORDER, next_stage_from_stages
from .bounded_graph import EvictionRecord, GraphBudget
from .config import DEFAULT_CONFIG, ENGINE_VERSION
from .evidence import apply_suppressions
from .features import FEATURE_NAMES, project_event
from .graph import _DELTA_FEATURE_INDEX, _normalize_delta, degree_normalize
from .models import CampaignModel, SelectiveSSM
from .pipeline import IntelligenceFinding, psi_drift
from .registry import feature_schema_hash
from .schema import KIND_PRIOR, SEVERITY_WEIGHT, IdrEvent
from .streaming import EVIDENCE_LIMIT

EXPORT_FORMAT = "idr-intelligence-onnx-v1"
OPSET = 18


class _StepCell(nn.Module):
    """Traceable wrapper over SelectiveSSM.step — one event's state advance."""

    def __init__(self, ssm: SelectiveSSM, time_aware: bool) -> None:
        super().__init__()
        self.ssm = ssm
        self.time_aware = time_aware

    def forward(
        self, x_t: torch.Tensor, state: torch.Tensor, output: torch.Tensor, delta_t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_aware:
            return self.ssm.step(x_t, state, output, delta_t=delta_t)
        return self.ssm.step(x_t, state, output)


class _RelationalHead(nn.Module):
    """Traceable wrapper: carried S6 outputs + normalized adjacency -> logits.

    active is built from ones_like so the node axis stays symbolic in the
    exported graph; every carried entity is active by construction, exactly as
    in StreamingScorer.finding().
    """

    def __init__(self, model: CampaignModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, outputs: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.model.temporal is not None
        node_state = self.model.temporal.norm(outputs).unsqueeze(0)
        active = torch.ones_like(outputs[:, :1]).unsqueeze(0)
        result = self.model.relational_head(node_state, active, adjacency.unsqueeze(0))
        return result.graph_logit, result.node_logits[0]


def export_streaming_bundle(model: CampaignModel, out_dir: str | Path, model_version: str = "development") -> dict[str, Any]:
    """Write step.onnx, head.onnx, and manifest.json for a streaming S6 model.

    Returns the manifest dict. Only S6 models are exportable — mirroring
    StreamingScorer, the static ablation has no carried state to step.
    """
    if model.temporal is None:
        raise ValueError("export requires an S6 model (use_s6=True); the static baseline has no step cell to export")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    time_aware = model.time_mode == "time_aware"

    step_inputs = ["x_t", "state", "output"] + (["delta_t"] if time_aware else [])
    step_args: tuple[torch.Tensor, ...] = (
        torch.zeros(1, model.feature_dim),
        torch.zeros(1, model.hidden_dim, model.state_dim),
        torch.zeros(1, model.hidden_dim),
    )
    if time_aware:
        step_args = (*step_args, torch.zeros(1))
    example_nodes = 3
    head_args = (torch.zeros(example_nodes, model.hidden_dim), torch.eye(example_nodes))

    with warnings.catch_warnings():
        # The TorchScript exporter is pinned deliberately: its op choices are
        # what the parity suite (onnxruntime + the Rust tract runtime) validates.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            _StepCell(model.temporal, time_aware),
            step_args,
            str(out / "step.onnx"),
            input_names=step_inputs,
            output_names=["state_out", "output_out"],
            opset_version=OPSET,
            dynamo=False,
        )
        torch.onnx.export(
            _RelationalHead(model),
            head_args,
            str(out / "head.onnx"),
            input_names=["outputs", "adjacency"],
            output_names=["graph_logit", "node_logits"],
            dynamic_axes={"outputs": {0: "nodes"}, "adjacency": {0: "nodes", 1: "nodes"}, "node_logits": {0: "nodes"}},
            opset_version=OPSET,
            dynamo=False,
        )

    manifest: dict[str, Any] = {
        "format": EXPORT_FORMAT,
        "engine_version": ENGINE_VERSION,
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_schema_hash": feature_schema_hash(),
        "model": {
            "feature_dim": model.feature_dim,
            "hidden_dim": model.hidden_dim,
            "state_dim": model.state_dim,
            "time_mode": model.time_mode,
            "pooling": model.pooling,
            "use_gnn": model.use_gnn,
            "decay_half_life": model.decay_half_life,
        },
        "calibration": {
            "temperature": float(model.temperature.item()),
            "bias": float(model.cal_bias.item()),
            "label": model.calibration_label(),
        },
        "features": {
            "names": list(FEATURE_NAMES),
            "severity_weight": dict(SEVERITY_WEIGHT),
            "severity_default": 0.15,
            "kind_prior": dict(KIND_PRIOR),
            "kind_prior_default": 0.18,
            "delta_log_divisor": 12.0,
        },
        "attack": {"tactic_order": list(TACTIC_ORDER), "kind_to_attack": dict(KIND_TO_ATTACK)},
        "scoring": {"top_k_default": DEFAULT_CONFIG.scoring.top_k, "evidence_limit": EVIDENCE_LIMIT},
        "feature_stats": model.feature_stats,
        "graphs": {
            "step": {"file": "step.onnx", "inputs": step_inputs, "outputs": ["state_out", "output_out"]},
            "head": {"file": "head.onnx", "inputs": ["outputs", "adjacency"], "outputs": ["graph_logit", "node_logits"]},
        },
        "opset": OPSET,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class _OnnxEntityState:
    """Per-entity carried scan state in the ONNX runner — mirrors EntityState."""

    state: np.ndarray  # [1, hidden, state_dim]
    output: np.ndarray  # [1, hidden]
    last_seen: datetime
    event_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)


class OnnxStreamScorer:
    """StreamingScorer semantics over an exported bundle — torch-free.

    This is the reference implementation of the serving contract: everything a
    non-Python consumer (the Rust bridge) must reproduce is either read from
    manifest.json or is deterministic bookkeeping written out here. Divergence
    from StreamingScorer is a bug, and a parity test pins the two together.
    """

    def __init__(self, bundle_dir: str | Path, budget: GraphBudget | None = None) -> None:
        import onnxruntime

        bundle = Path(bundle_dir)
        self.manifest: dict[str, Any] = json.loads((bundle / "manifest.json").read_text())
        if self.manifest.get("format") != EXPORT_FORMAT:
            raise ValueError(f"unrecognized export bundle format: {self.manifest.get('format')!r}")
        graphs = self.manifest["graphs"]
        self._step = onnxruntime.InferenceSession(str(bundle / graphs["step"]["file"]))
        self._head = onnxruntime.InferenceSession(str(bundle / graphs["head"]["file"]))
        self._step_inputs = graphs["step"]["inputs"]
        self.time_mode: str = self.manifest["model"]["time_mode"]
        self.decay_half_life: float | None = self.manifest["model"]["decay_half_life"]
        self.budget = budget
        self.entities: dict[str, _OnnxEntityState] = {}
        self.evictions: list[EvictionRecord] = []
        self.relation_counts: dict[str, int] = {}
        self._edge_last_seen: dict[tuple[str, str], datetime] = {}
        self._stages: dict[str, dict[str, Any]] = {}
        self._previous_time: datetime | None = None
        self._first_event: tuple[datetime, str] | None = None
        self._drift_counts: np.ndarray | None = None
        stats = self.manifest.get("feature_stats")
        if stats:
            self._drift_counts = np.zeros((self.manifest["model"]["feature_dim"], len(stats["bin_edges"]) - 1), dtype=np.int64)
        self.events_seen = 0

    def ingest(self, event: IdrEvent) -> None:
        """Advance every entity the event mentions by one exported-cell step."""
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
        hidden = self.manifest["model"]["hidden_dim"]
        state_dim = self.manifest["model"]["state_dim"]
        for entity in projection.entities:
            existing = self.entities.get(entity)
            gap_seconds = (event.timestamp - existing.last_seen).total_seconds() if existing is not None else 0.0
            entity_delta = _normalize_delta(gap_seconds)
            if self.time_mode == "global":
                features = projection.features
            else:
                features = projection.features.copy()
                features[_DELTA_FEATURE_INDEX] = entity_delta
            if existing is None:
                existing = _OnnxEntityState(
                    state=np.zeros((1, hidden, state_dim), dtype=np.float32),
                    output=np.zeros((1, hidden), dtype=np.float32),
                    last_seen=event.timestamp,
                )
                self.entities[entity] = existing
            feeds = {"x_t": features[np.newaxis, :], "state": existing.state, "output": existing.output}
            if "delta_t" in self._step_inputs:
                feeds["delta_t"] = np.array([entity_delta], dtype=np.float32)
            existing.state, existing.output = self._step.run(None, feeds)
            existing.last_seen = event.timestamp
            existing.event_count += 1
            if event.id not in existing.evidence_ids:
                existing.evidence_ids.append(event.id)
                del existing.evidence_ids[:-EVIDENCE_LIMIT]
            if self._drift_counts is not None:
                edges = np.asarray(self.manifest["feature_stats"]["bin_edges"])
                for index in range(features.shape[0]):
                    self._drift_counts[index] += np.histogram(features[index : index + 1], bins=edges)[0]
        for left, right, relation in projection.edges:
            self._edge_last_seen[(min(left, right), max(left, right))] = event.timestamp
            self.relation_counts[relation] = self.relation_counts.get(relation, 0) + 1
        self.events_seen += 1
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        if self.budget is None or len(self.entities) <= self.budget.max_nodes:
            return
        kept, evicted = self.budget.apply({entity: state.last_seen for entity, state in self.entities.items()})
        keep = set(kept)
        self.evictions.extend(evicted)
        for record in evicted:
            del self.entities[record.entity]
        self._edge_last_seen = {pair: seen for pair, seen in self._edge_last_seen.items() if pair[0] in keep and pair[1] in keep}

    def finding(self, top_k: int | None = None, suppressions: list[str] | None = None) -> IntelligenceFinding:
        """Score the carried state through the exported head graph."""
        if not self.entities:
            raise ValueError("no events ingested")
        if top_k is None:
            top_k = int(self.manifest["scoring"]["top_k_default"])
        node_ids = tuple(self.entities)
        index_of = {entity: index for index, entity in enumerate(node_ids)}
        adjacency = np.eye(len(node_ids), dtype=np.float32)
        assert self._previous_time is not None
        for (left, right), seen in self._edge_last_seen.items():
            weight = np.float32(1.0)
            if self.decay_half_life is not None:
                age = (self._previous_time - seen).total_seconds()
                weight = np.float32(0.5 ** (age / self.decay_half_life))
            i, j = index_of[left], index_of[right]
            adjacency[i, j] = adjacency[j, i] = weight
        adjacency = degree_normalize(adjacency)
        outputs = np.concatenate([state.output for state in self.entities.values()], axis=0)
        graph_logit, node_logits = self._head.run(None, {"outputs": outputs, "adjacency": adjacency})
        calibration = self.manifest["calibration"]
        raw_probability = float(_sigmoid(graph_logit[0]))
        probability = float(_sigmoid(graph_logit[0] / max(calibration["temperature"], 1e-3) + calibration["bias"]))
        node_probability = _sigmoid(node_logits)
        ranking_scores, applied_suppressions = apply_suppressions(node_ids, node_probability, suppressions or [])
        # Stable sort so exact ties rank in first-seen order — see score_events.
        ranked = np.argsort(-ranking_scores, kind="stable")[: min(top_k, len(node_ids))]
        ranked = np.array([index for index in ranked if np.isfinite(ranking_scores[index])], dtype=int)
        related = tuple(node_ids[index] for index in ranked)
        evidence = tuple(dict.fromkeys(event_id for index in ranked for event_id in self.entities[node_ids[index]].evidence_ids))
        stages = tuple(self._stages.values())
        assert self._first_event is not None
        drift = None
        if self._drift_counts is not None:
            drift = psi_drift(self.manifest["feature_stats"], self._drift_counts)
        return IntelligenceFinding(
            campaign_id=f"idr-campaign-{self._first_event[1][:8]}",
            escalation_probability=round(probability, 6),
            raw_escalation_probability=round(raw_probability, 6),
            calibration=calibration["label"],
            predicted_next_stage=next_stage_from_stages(stages),
            observed_attack_stages=stages,
            related_entities=related,
            entity_evidence=(),
            applied_suppressions=applied_suppressions,
            evidence_event_ids=evidence,
            model_version=str(self.manifest["model_version"]),
            graph_nodes=len(node_ids),
            graph_relations=dict(self.relation_counts),
            engine_version=str(self.manifest["engine_version"]),
            feature_schema_hash=str(self.manifest["feature_schema_hash"]),
            scored_at=datetime.now(UTC).isoformat(),
            feature_drift=drift,
            continues_campaign=False,
            windows_observed=1,
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
