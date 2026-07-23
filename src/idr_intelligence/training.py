"""Chronological four-way ablation benchmark over simulated campaigns."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn

from .config import DEFAULT_CONFIG
from .features import FEATURE_DIM
from .graph import build_temporal_graph
from .models import CampaignModel, save_checkpoint
from .simulator import simulate_campaign

HIDDEN_DIM = DEFAULT_CONFIG.model.hidden_dim
STATE_DIM = DEFAULT_CONFIG.model.state_dim


@dataclass(frozen=True)
class Batch:
    """Padded tensors for a set of campaign graphs plus their labels."""

    sequences: torch.Tensor
    mask: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor


def set_seed(seed: int) -> None:
    """Seed every RNG in play and pin Torch to one CPU thread."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_dataset(samples: int = 80, seed: int = 7, max_nodes: int = DEFAULT_CONFIG.graph.train_max_nodes, max_steps: int = DEFAULT_CONFIG.graph.train_max_steps) -> Batch:
    """Simulate alternating benign/malicious campaigns and pad them into one Batch."""
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
    """Split 60/20/20 by sample index, which the simulator orders by start time."""
    size = len(batch.labels)
    train_end, validation_end = int(size * 0.60), int(size * 0.80)
    return _slice(batch, 0, train_end), _slice(batch, train_end, validation_end), _slice(batch, validation_end, size)


def train_ablation(samples: int = 80, epochs: int = 3, seed: int = 7, output: str | None = None) -> dict:
    """Train all four variants under one split, save the hybrid, return the report."""
    set_seed(seed)
    train, validation, test = chronological_split(make_dataset(samples=samples, seed=seed))
    variants: dict[str, dict[str, Any]] = {
        "static_baseline": {"use_s6": False, "use_gnn": False},
        "s6_only": {"use_s6": True, "use_gnn": False},
        "gnn_only": {"use_s6": False, "use_gnn": True},
        "s6_gnn": {"use_s6": True, "use_gnn": True},
        "s6_gnn_uniform_pool": {"use_s6": True, "use_gnn": True, "pooling": "uniform"},
    }
    evidence_seeds = [seed + samples + offset for offset in range(8)]
    results, trained, evidence_precision = {}, {}, {}
    for name, kwargs in variants.items():
        model = CampaignModel(FEATURE_DIM, hidden_dim=HIDDEN_DIM, state_dim=STATE_DIM, **kwargs)
        _fit(model, train, validation, epochs)
        trained[name] = model
        results[name] = _evaluate(model, test)
        evidence_precision[name] = evidence_precision_at_5(model, evidence_seeds)
    report = {
        "problem": "predict campaign escalation from ordered IdrEvent histories and entity relationships",
        "split": {"train": len(train.labels), "validation": len(validation.labels), "test": len(test.labels)},
        "metrics": results,
        "evidence_precision_at_5": evidence_precision,
        "best_model": max(results, key=lambda name: results[name]["pr_auc"]),
        "warning": "Synthetic benchmark demonstrates the pipeline; it is not evidence of production accuracy.",
    }
    artifact = Path("artifacts/hybrid_model.pt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(trained["s6_gnn"], artifact)
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
    _fit_temperature(model, validation)


def _fit_temperature(model: CampaignModel, validation: Batch) -> None:
    """Grid-fit the temperature buffer on validation NLL.

    The grid includes T=1.0, so post-scaling validation NLL can never exceed
    the unscaled NLL — that hard guarantee is what CI asserts.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(validation.sequences, validation.mask, validation.adjacency).graph_logit
        candidates = torch.cat([torch.tensor([1.0]), torch.logspace(-0.7, 0.7, 29)])
        losses = [loss_fn(logits / t, validation.labels).item() for t in candidates]
    model.temperature.fill_(float(candidates[int(np.argmin(losses))]))


def evidence_precision_at_5(model: CampaignModel, seeds: list[int], k: int = 5) -> float:
    """Fraction of top-k ranked entities that are core (multi-event) campaign entities.

    In a malicious simulation the converged host/hash/IP/process touch several
    events while peripheral entities appear once, so multi-event membership is
    usable ground truth for the ranking the finding hands to idr-sentinel.
    """
    model.eval()
    precisions = []
    for seed in seeds:
        graph = build_temporal_graph(simulate_campaign(1, seed))
        with torch.no_grad():
            output = model(
                torch.from_numpy(graph.sequences).unsqueeze(0),
                torch.from_numpy(graph.mask).unsqueeze(0),
                torch.from_numpy(graph.adjacency).unsqueeze(0),
            )
        top = torch.argsort(output.node_logits[0], descending=True)[: min(k, graph.node_count)].tolist()
        core = {index for index, event_ids in enumerate(graph.evidence_ids) if len(event_ids) >= 2}
        precisions.append(len(core.intersection(top)) / len(top))
    return round(float(np.mean(precisions)), 6)


def _evaluate(model: CampaignModel, batch: Batch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(batch.sequences, batch.mask, batch.adjacency).graph_logit
        probability = model.calibrated_probability(logits).numpy()
    labels = batch.labels.numpy()
    ece, mce = _calibration_errors(labels, probability)
    return {
        "roc_auc": round(float(roc_auc_score(labels, probability)), 6),
        "pr_auc": round(float(average_precision_score(labels, probability)), 6),
        "brier": round(float(brier_score_loss(labels, probability)), 6),
        "log_loss": round(float(log_loss(labels, probability, labels=[0.0, 1.0])), 6),
        "ece": round(ece, 6),
        "max_calibration_error": round(mce, 6),
        "temperature": round(float(model.temperature.item()), 6),
    }


def _calibration_errors(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> tuple[float, float]:
    """Expected and max calibration error over equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    expected, maximum = 0.0, 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probability > low) & (probability <= high) if low > 0 else (probability >= low) & (probability <= high)
        if not selected.any():
            continue
        gap = abs(float(probability[selected].mean()) - float(labels[selected].mean()))
        expected += float(selected.mean()) * gap
        maximum = max(maximum, gap)
    return expected, maximum


def _slice(batch: Batch, start: int, end: int) -> Batch:
    return Batch(batch.sequences[start:end], batch.mask[start:end], batch.adjacency[start:end], batch.labels[start:end])
