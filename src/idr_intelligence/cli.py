"""Command-line entry points: synthetic demo and NDJSON scoring."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .campaigns import CampaignRegistry
from .models import CampaignModel, load_campaign_model, save_checkpoint
from .observability import LEVELS, configure_logging, get_logger, log_event
from .pipeline import IntelligenceFinding, score_events
from .registry import feature_schema_hash
from .schema import IdrEvent
from .simulator import SCENARIOS, simulate_campaign
from .training import (
    decay_ablation,
    rolling_origin_ablation,
    time_ablation,
    train_ablation,
)


def main() -> None:
    """Parse arguments and dispatch to the demo or score command."""
    parser = argparse.ArgumentParser(prog="idr-intelligence")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default="warning",
        choices=LEVELS,
        help="operational logging to stderr as JSON lines (default warning keeps the CLI quiet); "
        "logs never touch stdout, so the finding JSON is unchanged",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", parents=[common], help="run ablations and emit a synthetic campaign finding")
    demo.add_argument("--samples", type=int, default=80)
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--malicious-rate", type=float, default=0.5)
    demo.add_argument("--scenario", default="v0_easy", choices=SCENARIOS)
    demo.add_argument("--data", default=None, help="directory or file of *.labeled.ndjson windows; replaces the simulator")
    demo.add_argument("--output", default="reports/demo.json")

    score = subparsers.add_parser("score", parents=[common], help="score newline-delimited IdrEvent JSON")
    score.add_argument("events")
    score.add_argument("--weights", default="artifacts/hybrid_model.pt")
    score.add_argument("--suppress", action="append", default=None, help="entity id or 'prefix:' to attenuate from ranking (repeatable)")
    score.add_argument("--registry", default=None, help="campaign registry JSON path; matched and updated so campaign ids stay stable across windows")

    stream = subparsers.add_parser("stream", parents=[common], help="score newline-delimited IdrEvent JSON one event at a time over carried S6 state")
    stream.add_argument("events")
    stream.add_argument("--weights", default="artifacts/hybrid_model.pt")
    stream.add_argument("--max-nodes", type=int, default=4096, help="entity budget; least-recently-seen entities are evicted with an audit trail. 0 disables the bound. The default keeps the dense N x N scoring adjacency bounded on untrusted streams (mirrored by the Rust bridge CLI)")
    stream.add_argument("--suppress", action="append", default=None, help="entity id or 'prefix:' to attenuate from ranking (repeatable)")
    stream.add_argument("--registry", default=None, help="campaign registry JSON path; matched and updated so campaign ids stay stable across windows")

    export = subparsers.add_parser("export", parents=[common], help="export the streaming model as an ONNX bundle (step + head + manifest)")
    export.add_argument("--weights", default="artifacts/hybrid_model.pt")
    export.add_argument("--out", default="artifacts/export", help="output directory for step.onnx, head.onnx, manifest.json")
    export.add_argument("--model-version", default=None, help="model_version recorded in the manifest; defaults to the weights filename")

    validate = subparsers.add_parser("validate", parents=[common], help="real-data go/no-go gate: recalibrate + pick a threshold on a temporal holdout, snapshot real drift, emit a report + model card")
    validate.add_argument("--data", required=True, help="directory or file of *.labeled.ndjson real campaign windows")
    validate.add_argument("--weights", default="artifacts/hybrid_model.pt")
    validate.add_argument("--target-fpr", type=float, default=0.01, help="tolerated false-positive rate the operating threshold is chosen for")
    validate.add_argument("--holdout", type=float, default=0.5, help="fraction (latest by time) held out as the untouched test segment")
    validate.add_argument("--data-provenance", default="unspecified", help="operator attestation of the data source; the verdict is BINDING only when this names real data (not synthetic/sim/standin/demo/test)")
    validate.add_argument("--gate", action="append", default=None, metavar="KEY=VALUE", help="override a gate threshold, e.g. --gate max_ece=0.05 (repeatable); these are the operator's risk appetite")
    validate.add_argument("--out-weights", default=None, help="write the recalibrated checkpoint (real calibration + real drift baseline) here")
    validate.add_argument("--model-card", default=None, help="write a model card markdown here")

    bench = subparsers.add_parser("benchmark", parents=[common], help="run a frozen benchmark manifest; exit 1 on floor violations")
    bench.add_argument("--manifest", default="benchmarks/v1.json")

    ablation = subparsers.add_parser("ablation", parents=[common], help="rolling-origin CV with seed replicates; declares best_model or tie")
    ablation.add_argument("--samples", type=int, default=60)
    ablation.add_argument("--epochs", type=int, default=2)
    ablation.add_argument("--folds", type=int, default=3)
    ablation.add_argument("--replicates", type=int, default=3)
    ablation.add_argument("--malicious-rate", type=float, default=0.5)
    ablation.add_argument("--scenario", default="v0_easy", choices=SCENARIOS)

    timeabl = subparsers.add_parser("time-ablation", parents=[common], help="compare global / per-entity / time-aware S6 on one scenario")
    timeabl.add_argument("--scenario", default="low_and_slow", choices=SCENARIOS)
    timeabl.add_argument("--samples", type=int, default=80)
    timeabl.add_argument("--epochs", type=int, default=3)

    decayabl = subparsers.add_parser("decay-ablation", parents=[common], help="compare edge-decay half-lives (none / 1h / 15m) on one scenario")
    decayabl.add_argument("--scenario", default="distractor", choices=SCENARIOS)
    decayabl.add_argument("--samples", type=int, default=80)
    decayabl.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()
    configure_logging(args.log_level)
    log = get_logger("cli")
    log_event(log, "command_start", command=args.command, engine_schema=feature_schema_hash())
    if args.command == "demo":
        report = train_ablation(samples=args.samples, epochs=args.epochs, output=args.output, malicious_rate=args.malicious_rate, scenario=args.scenario, data=args.data)
        model = load_campaign_model("artifacts/hybrid_model.pt")
        finding = score_events(simulate_campaign(1, 999), model, model_version="synthetic-demo-v0.1")
        print(json.dumps({"benchmark": report, "finding": finding.to_dict()}, indent=2))
    elif args.command == "benchmark":
        result = run_benchmark(args.manifest)
        print(json.dumps({key: result[key] for key in ("suite_version", "passed", "violations")}, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "ablation":
        report = rolling_origin_ablation(
            samples=args.samples, epochs=args.epochs, folds=args.folds,
            replicates=args.replicates, malicious_rate=args.malicious_rate, scenario=args.scenario,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "time-ablation":
        print(json.dumps(time_ablation(scenario=args.scenario, samples=args.samples, epochs=args.epochs), indent=2))
    elif args.command == "decay-ablation":
        print(json.dumps(decay_ablation(scenario=args.scenario, samples=args.samples, epochs=args.epochs), indent=2))
    elif args.command == "validate":
        from .dataio import load_labeled_windows
        from .validation import render_model_card, validate_model

        model = _load_model(args.weights, log)
        windows = load_labeled_windows(args.data)
        gate_overrides = _parse_gate_overrides(args.gate)
        report = validate_model(model, windows, target_fpr=args.target_fpr, holdout=args.holdout, provenance=args.data_provenance, gates=gate_overrides)
        log_event(
            log, "validation_complete", verdict=report["verdict"], binding=report["binding"],
            windows=report["data"]["windows"], failed_gates=[gate["name"] for gate in report["gates"] if not gate["pass"]],
        )
        if args.model_card:
            Path(args.model_card).parent.mkdir(parents=True, exist_ok=True)
            Path(args.model_card).write_text(render_model_card(report))
            report["model_card"] = args.model_card
        # A safety gate only emits a deployable artifact when it passes.
        if args.out_weights and report["verdict"] == "go":
            save_checkpoint(model, args.out_weights)
            report["recalibrated_checkpoint"] = args.out_weights
        elif args.out_weights:
            report["recalibrated_checkpoint"] = None
            report["out_weights_withheld"] = "verdict is no-go; the recalibrated checkpoint was not written"
        print(json.dumps(report, indent=2))
        if report["verdict"] != "go":
            raise SystemExit(2)
    elif args.command == "export":
        from .export import export_streaming_bundle

        model = _load_model(args.weights, log)
        manifest = export_streaming_bundle(model, args.out, model_version=args.model_version or Path(args.weights).name)
        log_event(log, "bundle_exported", out=args.out, feature_schema_hash=manifest["feature_schema_hash"], graphs=sorted(graph["file"] for graph in manifest["graphs"].values()))
        print(json.dumps({
            "out": args.out,
            "graphs": sorted(graph["file"] for graph in manifest["graphs"].values()),
            "model": manifest["model"],
            "calibration": manifest["calibration"]["label"],
            "feature_schema_hash": manifest["feature_schema_hash"],
        }, indent=2))
    elif args.command == "stream":
        from .bounded_graph import GraphBudget
        from .streaming import StreamingScorer

        model = _load_model(args.weights, log)
        budget = GraphBudget(max_nodes=args.max_nodes) if args.max_nodes else None
        scorer = StreamingScorer(model, budget=budget, model_version=Path(args.weights).name)
        started = time.perf_counter()
        for event in sorted(_read_events(args.events, log), key=lambda item: (item.timestamp, item.id)):
            scorer.ingest(event)
        registry = CampaignRegistry.load(args.registry) if args.registry else None
        finding = scorer.finding(suppressions=args.suppress, registry=registry)
        if registry is not None:
            registry.save(args.registry)
        _log_finding(log, finding, elapsed_ms=_elapsed_ms(started), events=scorer.events_seen, evictions=len(scorer.evictions))
        payload = finding.to_dict()
        payload["evictions"] = [
            {"entity": record.entity, "last_seen": record.last_seen.isoformat(), "reason": record.reason}
            for record in scorer.evictions
        ]
        print(json.dumps(payload, indent=2))
    else:
        events = _read_events(args.events, log)
        model = _load_model(args.weights, log)
        registry = CampaignRegistry.load(args.registry) if args.registry else None
        started = time.perf_counter()
        finding = score_events(events, model, model_version=Path(args.weights).name, suppressions=args.suppress, registry=registry)
        if registry is not None:
            registry.save(args.registry)
        _log_finding(log, finding, elapsed_ms=_elapsed_ms(started), events=len(events))
        print(json.dumps(finding.to_dict(), indent=2))


def _load_model(path: str, log: Any) -> CampaignModel:
    """Load a checkpoint, logging its provenance — or a clean operational error."""
    try:
        model = load_campaign_model(path)
    except Exception as exc:
        log_event(log, "model_load_failed", level=logging.ERROR, weights=path, reason=str(exc))
        raise SystemExit(f"cannot load model {path}: {exc}") from exc
    log_event(
        log,
        "model_loaded",
        weights=path,
        feature_dim=model.feature_dim,
        time_mode=model.time_mode,
        decay_half_life=model.decay_half_life,
        calibration=model.calibration_label(),
        feature_schema_hash=feature_schema_hash(),
    )
    return model


def _log_finding(log: Any, finding: IntelligenceFinding, **fields: Any) -> None:
    """Emit the operational summary of a produced finding (never the full evidence)."""
    drift = finding.feature_drift or {}
    log_event(
        log,
        "finding_scored",
        campaign_id=finding.campaign_id,
        escalation_probability=finding.escalation_probability,
        calibration=finding.calibration,
        predicted_next_stage=finding.predicted_next_stage,
        graph_nodes=finding.graph_nodes,
        continues_campaign=finding.continues_campaign,
        drift_flagged=len(drift.get("flagged_features", ())),
        **fields,
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _parse_gate_overrides(raw: list[str] | None) -> dict[str, float] | None:
    """Parse repeated --gate KEY=VALUE flags into a float override map."""
    if not raw:
        return None
    overrides: dict[str, float] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--gate expects KEY=VALUE, got {item!r}")
        try:
            overrides[key.strip()] = float(value)
        except ValueError as exc:
            raise SystemExit(f"--gate {key}: {value!r} is not a number") from exc
    return overrides


def _read_events(path: str, log: Any = None) -> list[IdrEvent]:
    """Parse newline-delimited IdrEvent JSON, naming the offending line on failure."""
    events = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(IdrEvent.from_dict(json.loads(line)))
        except Exception as exc:
            if log is not None:
                log_event(log, "event_rejected", level=logging.ERROR, path=path, line=line_number, reason=str(exc))
            raise SystemExit(f"invalid event at line {line_number}: {exc}") from exc
    return events


if __name__ == "__main__":
    main()
