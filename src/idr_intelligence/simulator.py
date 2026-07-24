"""Matched benign/malicious campaign generator with graded and adversarial scenarios.

Every scenario emits the same six kill-chain kinds (plus interleaved noise for
`distractor`), so nothing separates the classes except convergence, order, and
attribute semantics — the exact signals the engine claims to use. All
construction is deterministic in (label, seed, scenario).

Scenario families:
- v0_easy:       the original all-or-nothing contrast (default; back-compatible)
- graded:        per-slot convergence with probability `difficulty`
- distractor:    converged campaign interleaved with benign noise on the same host
- legit_update:  hard negative — campaign-shaped topology with benign semantics
- truncated:     chronological prefix of the campaign; metadata carries stage_reached
- low_and_slow:  evasion — kill chain stretched from minutes to days
- split_host:    evasion — every stage on a different host, linked only via infrastructure
- hash_rotation: evasion — per-stage executable hashes on a converged host
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from .schema import IdrEvent

SCENARIOS = (
    "v0_easy",
    "graded",
    "distractor",
    "legit_update",
    "truncated",
    "low_and_slow",
    "split_host",
    "hash_rotation",
    "lateral_movement",
    "stale_preamble",
    "timing_only",
)


def simulate_campaign(
    label: int,
    seed: int,
    host: str | None = None,
    scenario: str = "v0_easy",
    difficulty: float = 0.7,
) -> list[IdrEvent]:
    """Emit one campaign's events for the given scenario, deterministically."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    primary_host = host or f"workstation-{seed % 19:02d}"
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=seed * 3)
    events: list[IdrEvent] = []

    def add(minutes: float, source: str, severity: str, kind: dict, event_host: str, stage_index: int) -> None:
        events.append(IdrEvent(
            id=str(uuid.UUID(int=(seed * 100 + len(events) + 1) % (1 << 128))),
            timestamp=start + timedelta(minutes=minutes),
            source=source,
            severity=severity,
            kind=kind,
            metadata={
                "host": event_host,
                "simulation_seed": seed,
                "scenario": scenario,
                "stage_index": stage_index,
            },
        ))

    suspicious_ip = f"203.0.113.{20 + seed % 70}"
    process_id = 7000 + seed
    campaign_hash = f"deadbeef{seed:056x}"[-64:]
    scatter_hosts = [
        primary_host,
        f"sensor-{seed % 7}",
        f"workstation-{(seed + 5) % 19:02d}",
        f"server-{seed % 11:02d}",
        f"laptop-{seed % 13:02d}",
        f"storage-{seed % 5:02d}",
    ]
    scatter_ips = [
        suspicious_ip,
        f"203.0.113.{(seed + 31) % 90 + 1}",
        f"203.0.113.{(seed + 47) % 90 + 1}",
        f"192.0.2.{(seed + 9) % 90 + 1}",
        f"198.51.100.{(seed + 55) % 90 + 1}",
    ]
    ordered_times: list[float] = [8, 10, 12, 17, 19, 23]
    shuffled_times: list[float] = [19, 8, 23, 10, 17, 12]

    converged = bool(label)
    hosts = [primary_host] * 6 if converged else list(scatter_hosts)
    ips = [suspicious_ip] * 5 if converged else list(scatter_ips)
    hashes = [campaign_hash] * 2 if converged else [campaign_hash, f"cafebabe{seed:056x}"[-64:]]
    pids = [process_id] * 2 if converged else [process_id, process_id + 900]
    times = list(ordered_times) if converged else list(shuffled_times)
    signed = False
    exfil = True
    hijack = True
    time_offset_seconds = 90.0

    if scenario == "graded" and label:
        threshold = int(difficulty * 100)
        hosts = [primary_host if (seed * 31 + slot) % 100 < threshold else scatter_hosts[slot] for slot in range(6)]
        ips = [suspicious_ip if (seed * 53 + slot) % 100 < threshold else scatter_ips[slot] for slot in range(5)]
        times = list(ordered_times) if (seed * 97) % 100 < threshold else list(shuffled_times)
    elif scenario == "legit_update" and not label:
        hosts = [primary_host] * 6
        ips = [suspicious_ip] * 5
        times = list(ordered_times)
        signed = True
        exfil = False
        hijack = False
        time_offset_seconds = 0.4
    elif scenario == "low_and_slow":
        times = [minute * 480 for minute in times]
    elif scenario == "split_host" and label:
        hosts = list(scatter_hosts)
    elif scenario == "hash_rotation" and label:
        hashes = [f"{stage:08x}{seed:056x}"[-64:] for stage in (0xA1, 0xA3)]
        pids = [process_id, process_id + 17]
    elif scenario == "lateral_movement" and label:
        # Scatter every infrastructure signal; only the shared user connects the stages.
        hosts = list(scatter_hosts)
        ips = list(scatter_ips)
        hashes = [campaign_hash, f"cafebabe{seed:056x}"[-64:]]
    elif scenario == "timing_only":
        # Identical benign content and converged structure for BOTH classes — the
        # ONLY discriminator is inter-event timing. Removes the graph-structure
        # crutch that saturates every other scenario, so temporal physics is the
        # sole path to separation. Malicious = tight burst; benign = routine
        # multi-hour cadence over the same entities.
        hosts = [primary_host] * 6
        ips = [suspicious_ip] * 5
        hashes = [campaign_hash] * 2
        pids = [process_id] * 2
        signed = True
        exfil = False
        hijack = False
        time_offset_seconds = 0.4
        times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0] if label else [0.0, 480.0, 960.0, 1440.0, 1920.0, 2400.0]

    add(times[0], "kernel_ebpf", "HIGH", {
        "type": "socket_lineage", "pid": pids[0], "tgid": pids[0],
        "exe_path": "/usr/bin/updater" if signed else "/tmp/.cache/update",
        "exe_sha256": hashes[0],
        "dst_ip": ips[0], "dst_port": 443, "is_signed": signed,
    }, hosts[0], 0)
    add(times[1], "sentinel_correlation", "HIGH", {
        "type": "bgp_anomaly",
        "kind": {
            "kind": "subprefix_hijack_local_infra" if hijack else "route_leak_benign",
            "covered_local_prefix": "203.0.113.0/24",
            "hijacker_asn": 64580 if hijack else 64500,
        },
        "prefix": "203.0.113.0/25", "observed_origin_asn": 64580 if hijack else 64500,
        "legitimate_origin_asn": 64500, "confidence": "high" if hijack else "low", "dst_ip": ips[1],
    }, hosts[1], 1)
    add(times[2], "kernel_ebpf", "CRITICAL", {
        "type": "suspicious_beacon", "pid": pids[1],
        "exe_path": "/usr/bin/updater" if signed else "/tmp/.cache/update",
        "exe_sha256": hashes[1],
        "dst_ip": ips[2], "asn_owner": "Example Transit",
    }, hosts[2], 2)
    add(times[3], "network_zeek", "HIGH", {
        "type": "ntp_time_shift", "offset_seconds": time_offset_seconds, "ntp_server": ips[3],
    }, hosts[3], 3)
    add(times[4], "network_zeek", "CRITICAL", {
        "type": "hsts_time_manipulation", "domain": "update-cdn.example",
        "cert_expiry": "2025-12-01T00:00:00Z", "ntp_shift_seconds": time_offset_seconds, "dst_ip": ips[4],
    }, hosts[4], 4)
    add(times[5], "hardware_nvme", "HIGH", {
        "type": "nvme_latency_anomaly", "device": "nvme0n1", "baseline_us": 120,
        "observed_us": 1900 + seed, "deviation_pct": 1483.0 if exfil else 40.0, "concurrent_exfil": exfil,
    }, hosts[5], 5)

    if scenario == "distractor":
        noise_times = [9, 14, 20] if label else [11, 16, 21]
        benign_hash = f"0badf00d{seed:056x}"[-64:]
        for noise_index, minute in enumerate(noise_times):
            add(minute, "kernel_ebpf", "INFO", {
                "type": "socket_lineage", "pid": process_id + 300 + noise_index,
                "tgid": process_id + 300 + noise_index,
                "exe_path": "/usr/bin/browser", "exe_sha256": benign_hash,
                "dst_ip": f"198.51.100.{(seed + noise_index * 13) % 90 + 1}", "dst_port": 443,
                "is_signed": True,
            }, primary_host, 6 + noise_index)

    if scenario == "lateral_movement":
        # Malicious: one actor across all hosts. Benign: a distinct actor per
        # host, so identity provides no cross-host link.
        shared_user = f"svc-account-{seed % 23:02d}"
        for event in events:
            event.kind["user"] = shared_user if label else f"local-user-{event.metadata['stage_index']}"
            event.kind["target_host"] = event.metadata["host"]

    if scenario == "stale_preamble":
        # Both classes carry an IDENTICAL benign preamble ~2 days before the
        # recent window (routine signed connections on the primary host). The
        # discriminating signal lives only in the recent burst, so the stale
        # preamble is pure noise that edge decay should suppress and per-entity
        # time should recognize as old — the case W11/W12 were built for.
        benign_hash = f"5afe0000{seed:056x}"[-64:]
        for pre_index in range(4):
            add(-2880.0 - pre_index * 180, "kernel_ebpf", "INFO", {
                "type": "socket_lineage", "pid": process_id - 500 - pre_index,
                "tgid": process_id - 500 - pre_index,
                "exe_path": "/usr/bin/routine", "exe_sha256": benign_hash,
                "dst_ip": f"198.51.100.{(seed + pre_index * 7) % 90 + 1}", "dst_port": 443,
                "is_signed": True,
            }, primary_host, 10 + pre_index)

    if scenario == "truncated":
        stage_reached = 2 + seed % 5
        events[:] = [event for event in events if event.metadata["stage_index"] < stage_reached]
        for event in events:
            event.metadata["stage_reached"] = stage_reached

    return sorted(events, key=lambda event: (event.timestamp, event.id))
