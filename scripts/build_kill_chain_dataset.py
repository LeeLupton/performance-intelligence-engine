"""Build a labeled multi-signal validation dataset from idr-main's own tools.

This is "option 1": generate a genuinely labeled, multi-modal
`LabeledWindow` dataset the engine can train and `validate` on, using the
detection platform's authoritative artifacts rather than the ML engine's own
simulator:

  positives (label 1): distinct campaigns instantiated from idr-main's
      `idr-sim full_kill_chain` template — the platform's own multi-modal
      kill-chain definition (kernel eBPF + network + hardware), varied per
      window (unique host, C2 IP, hash, pid, time) so each is a separate
      campaign, with malicious attributes (unsigned binary, concurrent exfil,
      large NTP shift, expired cert), interleaved with real BGP background.

  negatives (label 0): a mix of
      (a) benign-attribute variants of the SAME modalities (signed binary, no
          exfil, in-spec timing) — hard negatives that are structurally like a
          campaign but benign, so the model can't win by "has non-BGP events";
      (b) real idr-sentinel BGP anomaly clusters (the live feed is all
          `Observed*` calibration tier — genuinely benign, `production_alerts`
          is empty).

Honest provenance: the positive class and the benign hard-negatives are
INSTANTIATED FROM SIMULATED TEMPLATES (no confirmed real campaigns exist on the
platform — `production_alerts.json` is `[]`), while the easy negatives are real
telemetry. So `validate` will report a real multi-modal verdict but mark it
NON-BINDING — this harness becomes a binding pipeline the moment real labeled
positives (a red-team run driven through the live sentinel, or a confirmed
incident) replace the simulated ones.

Usage:
    python scripts/build_kill_chain_dataset.py \
        --template idrsim_killchain.ndjson \
        --anomalies /var/lib/idr-sentinel/state/anomalies.jsonl \
        --out data/killchain --positives 40 --negatives 40
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_sentinel_events as sentinel

_NS = uuid.UUID("6f4d9d2e-0000-4000-8000-000000000002")
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def load_template(path: Path) -> list[dict]:
    """Load the idr-sim kill-chain NDJSON template (the authoritative positive shape)."""
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not events:
        raise SystemExit(f"empty idr-sim template: {path}")
    return events


def _template_offsets(template: list[dict]) -> list[float]:
    """Seconds of each template event relative to the first (preserves idr-sim ordering)."""
    stamps = [datetime.fromisoformat(event["timestamp"]) for event in template]
    base = min(stamps)
    return [(stamp - base).total_seconds() for stamp in stamps]


def instantiate_campaign(template: list[dict], offsets: list[float], *, index: int, start: datetime, malicious: bool, rng: random.Random, stages: int | None = None) -> list[dict]:
    """One distinct campaign from the template: converge infra on a unique host, set
    malicious or benign attributes, and stamp a unique window time."""
    host = f"host-{index:04d}"
    c2_ip = f"198.51.{index % 256}.{10 + index % 200}"
    sha = f"{'de' if malicious else '5a'}{index:062x}"[-64:]
    pid = 4000 + index
    # A truncated campaign is one caught mid kill-chain: only the first `stages`
    # events fired. Realistic (early detection) and harder — fewer malicious
    # signals to converge on.
    keep = template if stages is None else template[:stages]
    keep_offsets = offsets if stages is None else offsets[:stages]
    events = []
    for event_index, (event, offset) in enumerate(zip(copy.deepcopy(keep), keep_offsets, strict=True)):
        kind = event["kind"]
        # Converge the campaign's infrastructure onto shared identifiers.
        for ip_key in ("src_ip", "dst_ip", "forward_ip", "reversed_ip", "ntp_server", "gateway_ip"):
            if ip_key in kind:
                kind[ip_key] = c2_ip
        if "exe_sha256" in kind:
            kind["exe_sha256"] = sha
        for pid_key in ("pid", "tgid"):
            if pid_key in kind:
                kind[pid_key] = pid
        # Malicious vs benign semantics on the exact features the model reads.
        if "is_signed" in kind:
            kind["is_signed"] = not malicious
        if "concurrent_exfil" in kind:
            kind["concurrent_exfil"] = malicious
        if "offset_seconds" in kind:
            kind["offset_seconds"] = 90.0 if malicious else 0.3
        if "ntp_shift_seconds" in kind:
            kind["ntp_shift_seconds"] = 90.0 if malicious else 0.3
        if "deviation_pct" in kind:
            kind["deviation_pct"] = 350.0 if malicious else 8.0
        if "observed_us" in kind:
            kind["observed_us"] = 450 if malicious else 110
        event["severity"] = event["severity"] if malicious else "INFO"
        stamp = start + timedelta(seconds=offset + rng.uniform(0.0, 0.5))
        event["timestamp"] = stamp.isoformat()
        event["metadata"] = {"host": host}
        event["id"] = str(uuid.uuid5(_NS, f"{index}|{malicious}|{event_index}|{kind.get('type')}"))
        events.append(event)
    return events


def bgp_background(prefix_records: list[dict], *, start: datetime, count: int) -> list[dict]:
    """A few real sentinel BGP anomalies as background, restamped into the window."""
    events = []
    for offset, record in enumerate(prefix_records[:count]):
        event = sentinel.to_idr_event(record)
        if event is None:
            continue
        event["timestamp"] = (start + timedelta(seconds=1.0 + offset)).isoformat()
        events.append(event)
    return events


def load_bgp_clusters(path: Path, *, scan_limit: int, want: int) -> list[list[dict]]:
    """Group real sentinel anomalies by prefix; return the largest coherent clusters."""
    if not path.is_file():
        return []
    clusters: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line_number >= scan_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prefix = record.get("prefix")
            if prefix:
                clusters.setdefault(prefix, []).append(record)
    ranked = sorted(clusters.values(), key=len, reverse=True)
    return [cluster for cluster in ranked if len(cluster) >= 3][:want]


def window_dict(window_id: str, label: int, events: list[dict]) -> dict:
    return {"window_id": window_id, "label": label, "events": sorted(events, key=lambda e: e["timestamp"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="idr-sim full_kill_chain NDJSON (the authoritative positive shape)")
    parser.add_argument("--anomalies", default="/var/lib/idr-sentinel/state/anomalies.jsonl")
    parser.add_argument("--out", default="data/killchain", help="output directory for the *.labeled.ndjson dataset")
    parser.add_argument("--positives", type=int, default=40)
    parser.add_argument("--negatives", type=int, default=40)
    parser.add_argument("--scan-limit", type=int, default=400000, help="bounded lines to scan from the anomaly log")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--truncate-frac", type=float, default=0.5, help="fraction of positives caught mid-chain (2-5 stages) — the hard, realistic case")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    template = load_template(Path(args.template))
    offsets = _template_offsets(template)
    # Real BGP clusters: enough for the easy-negative windows plus background.
    clusters = load_bgp_clusters(Path(args.anomalies), scan_limit=args.scan_limit, want=args.negatives + args.positives)

    total = args.positives + args.negatives
    # Interleave positives and negatives across time so every temporal segment
    # the validate gate carves out contains both classes.
    schedule = [1] * args.positives + [0] * args.negatives
    rng.shuffle(schedule)
    windows = []
    real_negatives = 0
    for index, label in enumerate(schedule):
        start = _EPOCH + timedelta(hours=index * 3)
        background = bgp_background(clusters[index], start=start, count=3) if index < len(clusters) else []
        if label == 1:
            stages = rng.randint(2, 5) if rng.random() < args.truncate_frac else None
            events = instantiate_campaign(template, offsets, index=index, start=start, malicious=True, rng=rng, stages=stages) + background
        elif index % 3 == 0 and index < len(clusters) and len(clusters[index]) >= 5:
            # Real all-negative BGP cluster as an easy, genuinely-real negative.
            events = bgp_background(clusters[index], start=start, count=min(len(clusters[index]), 24))
            real_negatives += 1
        else:
            # Benign multi-modal hard negative: same modalities, benign semantics.
            events = instantiate_campaign(template, offsets, index=index, start=start, malicious=False, rng=rng) + background
        windows.append(window_dict(f"w{index:04d}-{'mal' if label else 'ben'}", label, events))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = out / "campaigns.labeled.ndjson"
    dataset.write_text("\n".join(json.dumps(window) for window in windows) + "\n")
    print(json.dumps({
        "dataset": str(dataset),
        "windows": total,
        "positives": args.positives,
        "negatives": args.negatives,
        "real_bgp_negative_windows": real_negatives,
        "real_bgp_clusters_available": len(clusters),
        "modalities_in_positive": sorted({event["kind"]["type"] for event in template}),
        "provenance": "idrsim-killchain-positives+benign-variants+real-sentinel-bgp",
        "binding": False,
        "note": "Positives and benign hard-negatives are instantiated from idr-main's idr-sim template (simulated); "
                "easy negatives are real sentinel BGP. No confirmed real campaigns exist (production_alerts.json == []), "
                "so the validate verdict is intentionally non-binding until real labeled positives replace the simulated ones.",
    }, indent=2))


if __name__ == "__main__":
    main()
