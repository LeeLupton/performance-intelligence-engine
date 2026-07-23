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


class CampaignModel(nn.Module):
    """Ablatable S6 + GNN campaign classifier; either encoder can be disabled."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, state_dim: int = 8, use_s6: bool = True, use_gnn: bool = True) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.use_s6 = use_s6
        self.use_gnn = use_gnn
        self.temporal = SelectiveSSM(feature_dim, hidden_dim, state_dim) if use_s6 else None
        self.static = (
            None
            if use_s6
            else nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        )
        self.graph_layers = nn.ModuleList([ResidualGraphLayer(hidden_dim) for _ in range(2)]) if use_gnn else None
        self.node_head = nn.Linear(hidden_dim, 1)
        self.graph_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

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
        node_logits = self.node_head(node_state).squeeze(-1)
        active_nodes = (mask.sum(dim=-1) > 0).float().unsqueeze(-1)
        count = active_nodes.sum(dim=1).clamp_min(1.0)
        mean_pool = (node_state * active_nodes).sum(dim=1) / count
        max_pool = _masked_max(node_state, active_nodes, dim=1)
        graph_logit = self.graph_head(torch.cat([mean_pool, max_pool], dim=-1)).squeeze(-1)
        return ModelOutput(graph_logit=graph_logit, node_logits=node_logits)


def save_checkpoint(model: CampaignModel, path: str | Path) -> None:
    """Persist weights together with the dimensions needed to rebuild the model."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": model.feature_dim,
            "hidden_dim": model.hidden_dim,
            "state_dim": model.state_dim,
        },
        path,
    )


def load_campaign_model(path: str | Path) -> CampaignModel:
    """Rebuild a hybrid CampaignModel from a checkpoint written by save_checkpoint.

    Raw state dicts from pre-checkpoint versions are still accepted; their
    dimensions are inferred from tensor shapes, and weights of the formerly
    always-constructed (but unused) static branch are dropped.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
        model = CampaignModel(
            int(payload["feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            state_dim=int(payload["state_dim"]),
        )
        model.load_state_dict(payload["state_dict"])
    else:
        model = CampaignModel(
            payload["temporal.input_proj.weight"].shape[1],
            hidden_dim=payload["node_head.weight"].shape[1],
            state_dim=payload["temporal.a_log"].shape[1],
        )
        model.load_state_dict({key: value for key, value in payload.items() if not key.startswith("static.")})
    return model
