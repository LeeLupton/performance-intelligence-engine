from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

from .features import FEATURE_DIM
from .graph import build_temporal_graph
from .models import CampaignModel
from .simulator import simulate_campaign


@dataclass(frozen=True)
class Batch:
    sequences: torch.Tensor
    mask: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_dataset(samples: int = 80, seed: int = 7, max_nodes: int = 24, max_steps: int = 16) -> Batch:
    if samples < 20:
        raise ValueError("samples must be at least 20")
    sequences = np.zeros((samples, max_nodes, max_steps, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((samples, max_nodes, max_steps), dtype=np.float32)
    adjacency = np.zeros((samples, max_nodes, max_nodes), dtype=np.float32)
    labels = []
    for index in range(samples):
        label = int(index % 2 == 1)
        graph = build_temporal_graph(simulate_campaign(label, seed + index), max_steps=max_steps)
        count = min(graph.node_count, max_nodes)
        sequences[index, :count] = graph.sequences[:count]
        mask[index, :count] = graph.mask[:count]
        adjacency[index, :count, :count] = graph.adjacency[:count, :count]
        if count < max_nodes:
            adjacency[index, count:, count:] = np.eye(max_nodes - count, dtype=np.float32)
        labels.append(float(label))
    return Batch(torch.from_numpy(sequences), torch.from_numpy(mask), torch.from_numpy(adjacency), torch.tensor(labels))


def chronological_split(batch: Batch) -> tuple[Batch, Batch, Batch]:
    size = len(batch.labels)
    train_end, validation_end = int(size * 0.60), int(size * 0.80)
    return _slice(batch, 0, train_end), _slice(batch, train_end, validation_end), _slice(batch, validation_end, size)


def train_ablation(samples: int = 80, epochs: int = 3, seed: int = 7, output: str | None = None) -> dict:
    set_seed(seed)
    train, validation, test = chronological_split(make_dataset(samples=samples, seed=seed))
    variants = {
        "static_baseline": (False, False),
        "s6_only": (True, False),
        "gnn_only": (False, True),
        "s6_gnn": (True, True),
    }
    results, trained = {}, {}
    for name, (use_s6, use_gnn) in variants.items():
        model = CampaignModel(FEATURE_DIM, hidden_dim=24, state_dim=6, use_s6=use_s6, use_gnn=use_gnn)
        _fit(model, train, validation, epochs)
        trained[name] = model
        results[name] = _evaluate(model, test)
    report = {
        "problem": "predict campaign escalation from ordered IdrEvent histories and entity relationships",
        "split": {"train": len(train.labels), "validation": len(validation.labels), "test": len(test.labels)},
        "metrics": results,
        "best_model": max(results, key=lambda name: results[name]["pr_auc"]),
        "warning": "Synthetic benchmark demonstrates the pipeline; it is not evidence of production accuracy.",
    }
    artifact = Path("artifacts/hybrid_model.pt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained["s6_gnn"].state_dict(), artifact)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _fit(model: CampaignModel, train: Batch, validation: Batch, epochs: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(train.sequences, train.mask, train.adjacency).graph_logit, train.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(validation.sequences, validation.mask, validation.adjacency).graph_logit, validation.labels).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def _evaluate(model: CampaignModel, batch: Batch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(batch.sequences, batch.mask, batch.adjacency).graph_logit).numpy()
    labels = batch.labels.numpy()
    return {
        "roc_auc": round(float(roc_auc_score(labels, probability)), 6),
        "pr_auc": round(float(average_precision_score(labels, probability)), 6),
        "brier": round(float(brier_score_loss(labels, probability)), 6),
    }


def _slice(batch: Batch, start: int, end: int) -> Batch:
    return Batch(batch.sequences[start:end], batch.mask[start:end], batch.adjacency[start:end], batch.labels[start:end])
