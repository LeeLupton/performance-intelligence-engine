"""Temporal and relational encoders for campaign inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveSSM(nn.Module):
    """Auditable diagonal selective state-space layer."""

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim, state_dim))
        self.delta = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim * state_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim * state_dim)
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = self.input_proj(sequence)
        batch, steps, _ = u.shape
        state = torch.zeros(batch, self.hidden_dim, self.state_dim, device=u.device, dtype=u.dtype)
        output = torch.zeros(batch, self.hidden_dim, device=u.device, dtype=u.dtype)
        a = -torch.exp(self.a_log)
        for step in range(steps):
            current = u[:, step]
            active = mask[:, step].view(batch, 1, 1)
            delta = F.softplus(self.delta(current)).unsqueeze(-1).clamp(max=5.0)
            a_bar = torch.exp(delta * a)
            b = self.b_proj(current).view(batch, self.hidden_dim, self.state_dim)
            candidate = a_bar * state + delta * b * current.unsqueeze(-1)
            state = active * candidate + (1.0 - active) * state
            c = self.c_proj(current).view(batch, self.hidden_dim, self.state_dim)
            step_output = (c * state).sum(-1) + self.skip * current
            output = active.squeeze(-1) * step_output + (1.0 - active.squeeze(-1)) * output
        return self.norm(output)


class ResidualGraphLayer(nn.Module):
    """One round of degree-normalized message passing with a residual update."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.bmm(adjacency, nodes)
        update = F.gelu(self.self_linear(nodes) + self.neighbor_linear(neighbors))
        return self.norm(nodes + update)


@dataclass(frozen=True)
class ModelOutput:
    """Graph-level campaign logit plus per-node relevance logits."""

    graph_logit: torch.Tensor
    node_logits: torch.Tensor


def _masked_max(values: torch.Tensor, keep: torch.Tensor, dim: int) -> torch.Tensor:
    """Max over `dim` ignoring masked entries; rows with nothing kept become zeros."""
    maximum = values.masked_fill(keep == 0, float("-inf")).max(dim=dim).values
    return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))


class GatedAttentionPool(nn.Module):
    """Gated MIL attention whose scores drive both graph pooling and entity ranking.

    Because the pooled vector feeds the trained campaign head, gradient flows
    into the scores — unlike the former node_head, which the loss never reached.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, nodes: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.value(nodes)) * torch.sigmoid(self.gate(nodes))).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(active.squeeze(-1) == 0, float("-inf")), dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * nodes).sum(dim=1)
        return pooled, scores


class CampaignModel(nn.Module):
    """Ablatable S6 + GNN campaign classifier; either encoder and the pooling are swappable.

    pooling="attention" ranks entities with trained gated-attention scores;
    pooling="uniform" keeps the legacy mean+max pool with the (untrained)
    node_head ranking as an ablation arm.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, state_dim: int = 8, use_s6: bool = True, use_gnn: bool = True, pooling: str = "attention") -> None:
        super().__init__()
        if pooling not in ("attention", "uniform"):
            raise ValueError(f"unknown pooling mode: {pooling}")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.use_s6 = use_s6
        self.use_gnn = use_gnn
        self.pooling = pooling
        self.temporal = SelectiveSSM(feature_dim, hidden_dim, state_dim) if use_s6 else None
        self.static = (
            None
            if use_s6
            else nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        )
        self.graph_layers = nn.ModuleList([ResidualGraphLayer(hidden_dim) for _ in range(2)]) if use_gnn else None
        self.attention = GatedAttentionPool(hidden_dim) if pooling == "attention" else None
        self.node_head = None if pooling == "attention" else nn.Linear(hidden_dim, 1)
        self.graph_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.temperature: torch.Tensor
        self.cal_bias: torch.Tensor
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("cal_bias", torch.zeros(1))
        self.feature_stats: dict | None = None

    def calibrated_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Affine (Platt) calibrated probability: sigmoid(logit / T + bias).

        The bias term is load-bearing: training weights the loss by pos_weight,
        which shifts every logit by a constant log(w). A scale-only rescale
        (temperature alone) cannot remove a constant shift, so the bias is what
        lets calibrated probabilities track true frequencies under imbalance.
        Identity (T=1, bias=0) leaves logits untouched for uncalibrated models.
        """
        return torch.sigmoid(logits / self.temperature.clamp_min(1e-3) + self.cal_bias)

    def calibration_label(self) -> str:
        """Single source of truth for the calibration string in checkpoints and findings."""
        temperature = float(self.temperature.item())
        bias = float(self.cal_bias.item())
        if temperature == 1.0 and bias == 0.0:
            return "none"
        return f"affine:scale={1.0 / temperature:.6f},bias={bias:.6f}"

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor, adjacency: torch.Tensor) -> ModelOutput:
        batch, nodes, steps, features = sequences.shape
        flat_sequence = sequences.view(batch * nodes, steps, features)
        flat_mask = mask.view(batch * nodes, steps)
        if self.temporal is not None:
            node_state = self.temporal(flat_sequence, flat_mask)
        else:
            assert self.static is not None
            valid = flat_mask.unsqueeze(-1)
            count = valid.sum(dim=1).clamp_min(1.0)
            mean = (flat_sequence * valid).sum(dim=1) / count
            maximum = _masked_max(flat_sequence, valid, dim=1)
            node_state = self.static(torch.cat([mean, maximum], dim=-1))
        node_state = node_state.view(batch, nodes, -1)
        if self.graph_layers is not None:
            for layer in self.graph_layers:
                node_state = layer(node_state, adjacency)
        active_nodes = (mask.sum(dim=-1) > 0).float().unsqueeze(-1)
        count = active_nodes.sum(dim=1).clamp_min(1.0)
        mean_pool = (node_state * active_nodes).sum(dim=1) / count
        if self.attention is not None:
            pooled, node_logits = self.attention(node_state, active_nodes)
        else:
            assert self.node_head is not None
            node_logits = self.node_head(node_state).squeeze(-1)
            pooled = _masked_max(node_state, active_nodes, dim=1)
        graph_logit = self.graph_head(torch.cat([mean_pool, pooled], dim=-1)).squeeze(-1)
        return ModelOutput(graph_logit=graph_logit, node_logits=node_logits)


def save_checkpoint(model: CampaignModel, path: str | Path) -> None:
    """Persist weights, the exact model variant, and a provenance manifest."""
    from .registry import ModelManifest

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": model.feature_dim,
            "hidden_dim": model.hidden_dim,
            "state_dim": model.state_dim,
            "use_s6": model.use_s6,
            "use_gnn": model.use_gnn,
            "pooling": model.pooling,
            "manifest": ModelManifest.create(calibration=model.calibration_label()).to_dict(),
            "feature_stats": model.feature_stats,
        },
        path,
    )


def load_campaign_model(path: str | Path) -> CampaignModel:
    """Rebuild a CampaignModel from any checkpoint generation.

    Current checkpoints carry dims, ablation flags, and pooling mode. Raw state
    dicts from pre-checkpoint versions are still accepted: dimensions and flags
    are inferred from tensor shapes/keys, and weights of the formerly
    always-constructed (but unused) static branch are dropped for S6 models.
    """
    from .registry import ModelManifest

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"unrecognized checkpoint format in {path}: expected a dict payload")
    if "manifest" in payload:
        ModelManifest.from_dict(payload["manifest"]).verify_feature_schema()
    if "state_dict" in payload:
        state_dict = dict(payload["state_dict"])
        state_dict.setdefault("temperature", torch.ones(1))
        state_dict.setdefault("cal_bias", torch.zeros(1))
        model = CampaignModel(
            int(payload["feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            state_dim=int(payload["state_dim"]),
            use_s6=bool(payload.get("use_s6", True)),
            use_gnn=bool(payload.get("use_gnn", True)),
            pooling=str(payload.get("pooling", _infer_pooling(state_dict))),
        )
        model.load_state_dict(state_dict)
        model.feature_stats = payload.get("feature_stats")
        return model
    use_s6 = any(key.startswith("temporal.") for key in payload)
    if use_s6:
        feature_dim = payload["temporal.input_proj.weight"].shape[1]
        state_dim = payload["temporal.a_log"].shape[1]
        kept = {key: value for key, value in payload.items() if not key.startswith("static.")}
    else:
        feature_dim = payload["static.0.weight"].shape[1] // 2
        state_dim = 1
        kept = dict(payload)
    kept.setdefault("temperature", torch.ones(1))
    kept.setdefault("cal_bias", torch.zeros(1))
    model = CampaignModel(
        feature_dim,
        hidden_dim=payload["node_head.weight"].shape[1],
        state_dim=state_dim,
        use_s6=use_s6,
        use_gnn=any(key.startswith("graph_layers.") for key in payload),
        pooling="uniform",
    )
    model.load_state_dict(kept)
    return model


def _infer_pooling(state_dict: dict) -> str:
    """Checkpoints predating the pooling flag are uniform unless attention keys exist."""
    return "attention" if any(key.startswith("attention.") for key in state_dict) else "uniform"
