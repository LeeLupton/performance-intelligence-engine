"""Chronological four-way ablation benchmark over simulated campaigns."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from torch import nn

from .config import DEFAULT_CONFIG
from .features import FEATURE_DIM
from .graph import build_temporal_graph
from .models import CampaignModel, save_checkpoint
from .simulator import SCENARIOS, simulate_campaign

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


def make_dataset(samples: int = 80, seed: int = 7, max_nodes: int = DEFAULT_CONFIG.graph.train_max_nodes, max_steps: int = DEFAULT_CONFIG.graph.train_max_steps, malicious_rate: float = 0.5, scenario: str = "v0_easy") -> Batch:
    """Simulate benign/malicious campaigns with seeded random labels and pad into one Batch.

    Labels are drawn per-sample (not the former index%2 alternation, which
    leaked parity structure across chronological split boundaries), with a
    deterministic retry so every chronological segment contains both classes.
    """
    if samples < 20:
        raise ValueError("samples must be at least 20")
    if not 0.0 < malicious_rate < 1.0:
        raise ValueError("malicious_rate must be strictly between 0 and 1")
    label_row = _draw_labels(samples, seed, malicious_rate)
    sequences = np.zeros((samples, max_nodes, max_steps, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((samples, max_nodes, max_steps), dtype=np.float32)
    adjacency = np.zeros((samples, max_nodes, max_nodes), dtype=np.float32)
    labels = []
    for index in range(samples):
        label = int(label_row[index])
        graph = build_temporal_graph(simulate_campaign(label, seed + index, scenario=scenario), max_steps=max_steps)
        count = min(graph.node_count, max_nodes)
        sequences[index, :count] = graph.sequences[:count]
        mask[index, :count] = graph.mask[:count]
        adjacency[index, :count, :count] = graph.adjacency[:count, :count]
        if count < max_nodes:
            adjacency[index, count:, count:] = np.eye(max_nodes - count, dtype=np.float32)
        labels.append(float(label))
    return Batch(torch.from_numpy(sequences), torch.from_numpy(mask), torch.from_numpy(adjacency), torch.tensor(labels))


def windows_to_batch(windows: list, max_nodes: int = DEFAULT_CONFIG.graph.train_max_nodes, max_steps: int = DEFAULT_CONFIG.graph.train_max_steps) -> Batch:
    """Pad chronologically-sorted labeled windows into one Batch, zero model changes."""
    samples = len(windows)
    sequences = np.zeros((samples, max_nodes, max_steps, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((samples, max_nodes, max_steps), dtype=np.float32)
    adjacency = np.zeros((samples, max_nodes, max_nodes), dtype=np.float32)
    labels = []
    for index, window in enumerate(windows):
        graph = build_temporal_graph(list(window.events), max_steps=max_steps)
        count = min(graph.node_count, max_nodes)
        sequences[index, :count] = graph.sequences[:count]
        mask[index, :count] = graph.mask[:count]
        adjacency[index, :count, :count] = graph.adjacency[:count, :count]
        if count < max_nodes:
            adjacency[index, count:, count:] = np.eye(max_nodes - count, dtype=np.float32)
        labels.append(float(window.label))
    return Batch(torch.from_numpy(sequences), torch.from_numpy(mask), torch.from_numpy(adjacency), torch.tensor(labels))


def _require_two_class_segments(labels: torch.Tensor) -> None:
    """Real data cannot be redrawn: fail loudly when a split segment is single-class."""
    size = len(labels)
    train_end, validation_end = int(size * 0.60), int(size * 0.80)
    for name, start, end in (("train", 0, train_end), ("validation", train_end, validation_end), ("test", validation_end, size)):
        segment = labels[start:end]
        if segment.sum() == 0 or segment.sum() == len(segment):
            raise ValueError(
                f"chronological {name} segment is single-class ({int(segment.sum())}/{len(segment)} malicious); "
                "supply more windows or a wider time range"
            )


def _draw_labels(samples: int, seed: int, malicious_rate: float) -> np.ndarray:
    """Seeded label draw, deterministically retried until every split segment has both classes."""
    train_end, validation_end = int(samples * 0.60), int(samples * 0.80)
    segments = ((0, train_end), (train_end, validation_end), (validation_end, samples))
    rng = np.random.default_rng(seed + 104729)
    for _ in range(1000):
        labels = rng.random(samples) < malicious_rate
        if all(0 < labels[start:end].sum() < end - start for start, end in segments):
            return labels
    raise ValueError(f"could not draw two-class segments at malicious_rate={malicious_rate} with {samples} samples")


def chronological_split(batch: Batch) -> tuple[Batch, Batch, Batch]:
    """Split 60/20/20 by sample index, which the simulator orders by start time."""
    size = len(batch.labels)
    train_end, validation_end = int(size * 0.60), int(size * 0.80)
    return _slice(batch, 0, train_end), _slice(batch, train_end, validation_end), _slice(batch, validation_end, size)


def train_ablation(samples: int = 80, epochs: int = 3, seed: int = 7, output: str | None = None, malicious_rate: float = 0.5, scenario: str = "v0_easy", data: str | None = None) -> dict:
    """Train every ablation variant under one split, save the hybrid, return the report.

    With `data`, labeled real windows (*.labeled.ndjson) replace the simulator;
    windows are ordered by start time so the chronological split stays honest.
    """
    set_seed(seed)
    if data is not None:
        from .dataio import load_labeled_windows

        windows = load_labeled_windows(data)
        batch = windows_to_batch(windows)
        _require_two_class_segments(batch.labels)
        source = str(data)
        observed_rate = float(batch.labels.mean().item())
    else:
        batch = make_dataset(samples=samples, seed=seed, malicious_rate=malicious_rate, scenario=scenario)
        source = "simulator"
        observed_rate = malicious_rate
    train, validation, test = chronological_split(batch)
    evidence_seeds = [seed + samples + offset for offset in range(8)]
    results, trained, evidence_precision = {}, {}, {}
    for name, kwargs in VARIANTS.items():
        model = CampaignModel(FEATURE_DIM, hidden_dim=HIDDEN_DIM, state_dim=STATE_DIM, **kwargs)
        _fit(model, train, validation, epochs)
        trained[name] = model
        results[name] = _evaluate(model, test)
        evidence_precision[name] = evidence_precision_at_5(model, evidence_seeds)
    report = {
        "problem": "predict campaign escalation from ordered IdrEvent histories and entity relationships",
        "data_source": source,
        "malicious_rate": observed_rate,
        "scenario": scenario if data is None else "real_data",
        "split": {"train": len(train.labels), "validation": len(validation.labels), "test": len(test.labels)},
        "metrics": results,
        "evidence_precision_at_5": evidence_precision,
        "scenario_generalization": scenario_generalization(trained["s6_gnn"], seed=seed + samples + 1000),
        "best_model": max(results, key=lambda name: results[name]["pr_auc"]),
        "warning": "Synthetic benchmark demonstrates the pipeline; it is not evidence of production accuracy.",
    }
    trained["s6_gnn"].feature_stats = feature_snapshot(train)
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
    positives = train.labels.sum().clamp_min(1.0)
    negatives = (len(train.labels) - train.labels.sum()).clamp_min(1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=negatives / positives)
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
    _fit_calibration(model, validation)


def _fit_calibration(model: CampaignModel, validation: Batch) -> None:
    """Fit affine calibration (scale + bias) on validation NLL by LBFGS.

    Optimizing scale and bias generalizes temperature scaling: the bias absorbs
    the constant log(pos_weight) shift that class-weighted training introduces,
    which temperature alone cannot. The fit starts at identity and is accepted
    only if it does not worsen validation NLL, preserving the hard guarantee
    that post-calibration NLL never exceeds the unscaled NLL.
    """
    model.eval()
    with torch.no_grad():
        logits = model(validation.sequences, validation.mask, validation.adjacency).graph_logit
    scale, bias = affine_calibration_params(logits, validation.labels)
    model.temperature.fill_(1.0 / scale)
    model.cal_bias.fill_(bias)


def affine_calibration_params(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Fit (scale, bias) minimizing validation NLL of sigmoid(scale*logit + bias).

    Optimizes from identity by LBFGS and returns identity (1.0, 0.0) unless the
    fit strictly does not worsen NLL — so calibration can never degrade the
    held-out NLL relative to the raw logits. Scale and bias are bounded so a
    perfectly separable validation set cannot drive the fit to runaway
    overconfidence; the bounds match the old temperature grid's implied range.
    """
    scale_min, scale_max, bias_bound = 0.2, 5.0, 5.0
    loss_fn = nn.BCEWithLogitsLoss()
    logits = logits.detach()
    with torch.no_grad():
        baseline = loss_fn(logits, labels).item()
    log_scale = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_scale, bias], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits * torch.exp(log_scale) + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scale = min(max(float(torch.exp(log_scale)), scale_min), scale_max)
        fitted_bias = min(max(float(bias), -bias_bound), bias_bound)
        fitted = loss_fn(logits * scale + fitted_bias, labels).item()
    if fitted <= baseline:
        return scale, fitted_bias
    return 1.0, 0.0


DRIFT_BIN_EDGES = [round(edge, 2) for edge in np.linspace(0.0, 1.5, 11).tolist()]


def feature_snapshot(batch: Batch) -> dict:
    """Per-feature histograms over the training batch's real event rows, for drift checks."""
    rows = batch.sequences[batch.mask > 0].numpy()
    histograms = [np.histogram(rows[:, index], bins=DRIFT_BIN_EDGES)[0].tolist() for index in range(rows.shape[1])]
    return {"bin_edges": DRIFT_BIN_EDGES, "histograms": histograms, "sample_count": int(rows.shape[0])}


def scenario_generalization(model: CampaignModel, seed: int, samples: int = 20) -> dict[str, dict[str, float]]:
    """Evaluate one trained model across every scenario family on held-out seeds.

    Labels alternate — harmless here because these sets are never split — and
    seeds sit far above the training range. This is the table that makes
    evasion-scenario regressions visible: a v0_easy-trained model is expected
    to degrade on the evasion families until the modeling workstreams land.
    """
    rows: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        graphs = []
        labels = []
        for index in range(samples):
            label = index % 2
            graphs.append(build_temporal_graph(simulate_campaign(label, seed + index, scenario=scenario)))
            labels.append(float(label))
        max_nodes = max(graph.node_count for graph in graphs)
        max_steps = max(graph.sequences.shape[1] for graph in graphs)
        sequences = np.zeros((samples, max_nodes, max_steps, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((samples, max_nodes, max_steps), dtype=np.float32)
        adjacency = np.zeros((samples, max_nodes, max_nodes), dtype=np.float32)
        for index, graph in enumerate(graphs):
            count = graph.node_count
            sequences[index, :count, : graph.sequences.shape[1]] = graph.sequences
            mask[index, :count, : graph.sequences.shape[1]] = graph.mask
            adjacency[index, :count, :count] = graph.adjacency
            if count < max_nodes:
                adjacency[index, count:, count:] = np.eye(max_nodes - count, dtype=np.float32)
        batch = Batch(torch.from_numpy(sequences), torch.from_numpy(mask), torch.from_numpy(adjacency), torch.tensor(labels))
        metrics = _evaluate(model, batch)
        rows[scenario] = {key: metrics[key] for key in ("roc_auc", "brier", "recall_at_fpr_1pct")}
    return rows


VARIANTS: dict[str, dict[str, Any]] = {
    "static_baseline": {"use_s6": False, "use_gnn": False},
    "s6_only": {"use_s6": True, "use_gnn": False},
    "gnn_only": {"use_s6": False, "use_gnn": True},
    "s6_gnn": {"use_s6": True, "use_gnn": True},
    "s6_gnn_uniform_pool": {"use_s6": True, "use_gnn": True, "pooling": "uniform"},
}


def rolling_origin_ablation(
    samples: int = 60,
    epochs: int = 2,
    seed: int = 7,
    folds: int = 3,
    replicates: int = 3,
    malicious_rate: float = 0.5,
    scenario: str = "v0_easy",
) -> dict:
    """Expanding-window temporal CV with seed replicates over every variant.

    Decision metric is Brier (ranking saturates on synthetic data). best_model
    is declared only when the winner beats the runner-up by more than the
    paired per-fold std of their differences; otherwise the verdict is "tie" —
    with 16 test samples and one seed, a single-split winner is noise.
    """
    if folds not in (2, 3):
        raise ValueError("folds must be 2 or 3 (expanding windows of 15% with a 40% minimum origin)")
    fold_offsets = list(range(3 - folds, 3))
    scores: dict[str, list[float]] = {name: [] for name in VARIANTS}
    pr_scores: dict[str, list[float]] = {name: [] for name in VARIANTS}
    for replicate in range(replicates):
        replicate_seed = seed + replicate * 10_000
        set_seed(replicate_seed)
        batch = make_dataset(samples=samples, seed=replicate_seed, malicious_rate=malicious_rate, scenario=scenario)
        size = len(batch.labels)
        for offset in fold_offsets:
            train_end = int(size * (0.40 + offset * 0.15))
            validation_end = int(size * (0.55 + offset * 0.15))
            test_end = int(size * (0.70 + offset * 0.15))
            train = _slice(batch, 0, train_end)
            validation = _slice(batch, train_end, validation_end)
            test = _slice(batch, validation_end, test_end)
            if any(part.labels.sum() in (0, len(part.labels)) for part in (train, validation, test)):
                continue
            for name, kwargs in VARIANTS.items():
                model = CampaignModel(FEATURE_DIM, hidden_dim=HIDDEN_DIM, state_dim=STATE_DIM, **kwargs)
                _fit(model, train, validation, epochs)
                metrics = _evaluate(model, test)
                scores[name].append(metrics["brier"])
                pr_scores[name].append(metrics["pr_auc"])
    if not next(iter(scores.values())):
        raise ValueError("every fold was single-class; increase samples or folds")
    per_variant = {
        name: {
            "mean_brier": round(float(np.mean(values)), 6),
            "std_brier": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 6),
            "mean_pr_auc": round(float(np.mean(pr_scores[name])), 6),
            "folds_evaluated": len(values),
        }
        for name, values in scores.items()
    }
    ordered = sorted(per_variant, key=lambda name: per_variant[name]["mean_brier"])
    best, runner_up = ordered[0], ordered[1]
    paired = np.asarray(scores[runner_up]) - np.asarray(scores[best])
    margin = round(float(paired.mean()), 6)
    paired_std = round(float(paired.std(ddof=1)) if len(paired) > 1 else 0.0, 6)
    significant = margin > paired_std
    return {
        "metric": "brier",
        "folds": folds,
        "replicates": replicates,
        "per_variant": per_variant,
        "best_model": best if significant else "tie",
        "decision": {"winner": best, "runner_up": runner_up, "margin": margin, "paired_std": paired_std, "significant": significant},
        "warning": "Synthetic benchmark demonstrates the pipeline; it is not evidence of production accuracy.",
    }


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
        "recall_at_fpr_1pct": round(_recall_at_fpr(labels, probability, 0.01), 6),
        "recall_at_fpr_0p1pct": round(_recall_at_fpr(labels, probability, 0.001), 6),
        "precision_at_5": round(_precision_at_k(labels, probability, 5), 6),
    }


def _recall_at_fpr(labels: np.ndarray, probability: np.ndarray, max_fpr: float) -> float:
    """Highest TPR achievable while keeping FPR at or below max_fpr.

    Degenerate single-class inputs make the ROC axes undefined; return the
    only well-defined answer rather than a NaN that would poison the report:
    with no positives no recall is achievable (0.0); with no negatives every
    threshold has zero false positives, so recall 1.0 is achievable at any
    budget.
    """
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    if positives == len(labels):
        return 1.0
    fpr, tpr, _ = roc_curve(labels, probability)
    admissible = tpr[fpr <= max_fpr]
    return float(admissible.max()) if admissible.size else 0.0


def _precision_at_k(labels: np.ndarray, probability: np.ndarray, k: int) -> float:
    """Fraction of the k highest-scored samples that are truly malicious."""
    k = min(k, len(labels))
    top = np.argsort(-probability)[:k]
    return float(labels[top].mean())


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
