"""Command-line entry points: synthetic demo and NDJSON scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .campaigns import CampaignRegistry
from .models import load_campaign_model
from .pipeline import score_events
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run ablations and emit a synthetic campaign finding")
    demo.add_argument("--samples", type=int, default=80)
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--malicious-rate", type=float, default=0.5)
    demo.add_argument("--scenario", default="v0_easy", choices=SCENARIOS)
    demo.add_argument("--data", default=None, help="directory or file of *.labeled.ndjson windows; replaces the simulator")
    demo.add_argument("--output", default="reports/demo.json")

    score = subparsers.add_parser("score", help="score newline-delimited IdrEvent JSON")
    score.add_argument("events")
    score.add_argument("--weights", default="artifacts/hybrid_model.pt")
    score.add_argument("--suppress", action="append", default=None, help="entity id or 'prefix:' to attenuate from ranking (repeatable)")
    score.add_argument("--registry", default=None, help="campaign registry JSON path; matched and updated so campaign ids stay stable across windows")

    stream = subparsers.add_parser("stream", help="score newline-delimited IdrEvent JSON one event at a time over carried S6 state")
    stream.add_argument("events")
    stream.add_argument("--weights", default="artifacts/hybrid_model.pt")
    stream.add_argument("--max-nodes", type=int, default=None, help="entity budget; least-recently-seen entities are evicted with an audit trail")
    stream.add_argument("--suppress", action="append", default=None, help="entity id or 'prefix:' to attenuate from ranking (repeatable)")
    stream.add_argument("--registry", default=None, help="campaign registry JSON path; matched and updated so campaign ids stay stable across windows")

    bench = subparsers.add_parser("benchmark", help="run a frozen benchmark manifest; exit 1 on floor violations")
    bench.add_argument("--manifest", default="benchmarks/v1.json")

    ablation = subparsers.add_parser("ablation", help="rolling-origin CV with seed replicates; declares best_model or tie")
    ablation.add_argument("--samples", type=int, default=60)
    ablation.add_argument("--epochs", type=int, default=2)
    ablation.add_argument("--folds", type=int, default=3)
    ablation.add_argument("--replicates", type=int, default=3)
    ablation.add_argument("--malicious-rate", type=float, default=0.5)
    ablation.add_argument("--scenario", default="v0_easy", choices=SCENARIOS)

    timeabl = subparsers.add_parser("time-ablation", help="compare global / per-entity / time-aware S6 on one scenario")
    timeabl.add_argument("--scenario", default="low_and_slow", choices=SCENARIOS)
    timeabl.add_argument("--samples", type=int, default=80)
    timeabl.add_argument("--epochs", type=int, default=3)

    decayabl = subparsers.add_parser("decay-ablation", help="compare edge-decay half-lives (none / 1h / 15m) on one scenario")
    decayabl.add_argument("--scenario", default="distractor", choices=SCENARIOS)
    decayabl.add_argument("--samples", type=int, default=80)
    decayabl.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()
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
    elif args.command == "stream":
        from .bounded_graph import GraphBudget
        from .streaming import StreamingScorer

        model = load_campaign_model(args.weights)
        budget = GraphBudget(max_nodes=args.max_nodes) if args.max_nodes else None
        scorer = StreamingScorer(model, budget=budget, model_version=Path(args.weights).name)
        for event in sorted(_read_events(args.events), key=lambda item: (item.timestamp, item.id)):
            scorer.ingest(event)
        registry = CampaignRegistry.load(args.registry) if args.registry else None
        finding = scorer.finding(suppressions=args.suppress, registry=registry)
        if registry is not None:
            registry.save(args.registry)
        payload = finding.to_dict()
        payload["evictions"] = [
            {"entity": record.entity, "last_seen": record.last_seen.isoformat(), "reason": record.reason}
            for record in scorer.evictions
        ]
        print(json.dumps(payload, indent=2))
    else:
        events = _read_events(args.events)
        model = load_campaign_model(args.weights)
        registry = CampaignRegistry.load(args.registry) if args.registry else None
        finding = score_events(events, model, model_version=Path(args.weights).name, suppressions=args.suppress, registry=registry)
        if registry is not None:
            registry.save(args.registry)
        print(json.dumps(finding.to_dict(), indent=2))


def _read_events(path: str) -> list[IdrEvent]:
    """Parse newline-delimited IdrEvent JSON, naming the offending line on failure."""
    events = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(IdrEvent.from_dict(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"invalid event at line {line_number}: {exc}") from exc
    return events


if __name__ == "__main__":
    main()
