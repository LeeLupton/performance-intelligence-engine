from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .features import FEATURE_DIM
from .models import CampaignModel
from .pipeline import score_events
from .schema import IdrEvent
from .simulator import simulate_campaign
from .training import train_ablation


def main() -> None:
    parser = argparse.ArgumentParser(prog="idr-intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run ablations and emit a synthetic campaign finding")
    demo.add_argument("--samples", type=int, default=80)
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--output", default="reports/demo.json")

    score = subparsers.add_parser("score", help="score newline-delimited IdrEvent JSON")
    score.add_argument("events")
    score.add_argument("--weights", default="artifacts/hybrid_model.pt")

    args = parser.parse_args()
    if args.command == "demo":
        report = train_ablation(samples=args.samples, epochs=args.epochs, output=args.output)
        model = CampaignModel(FEATURE_DIM, hidden_dim=24, state_dim=6)
        model.load_state_dict(torch.load("artifacts/hybrid_model.pt", map_location="cpu", weights_only=True))
        finding = score_events(simulate_campaign(1, 999), model, model_version="synthetic-demo-v0.1")
        print(json.dumps({"benchmark": report, "finding": finding.to_dict()}, indent=2))
    else:
        events = []
        for line_number, line in enumerate(Path(args.events).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(IdrEvent.from_dict(json.loads(line)))
            except Exception as exc:
                raise SystemExit(f"invalid event at line {line_number}: {exc}") from exc
        model = CampaignModel(FEATURE_DIM, hidden_dim=24, state_dim=6)
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        print(json.dumps(score_events(events, model, model_version=Path(args.weights).name).to_dict(), indent=2))


if __name__ == "__main__":
    main()
