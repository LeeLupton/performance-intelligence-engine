"""Matched benign/malicious campaign generator for the synthetic benchmark."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from .schema import IdrEvent


def simulate_campaign(label: int, seed: int, host: str | None = None) -> list[IdrEvent]:
    """Emit six events; label=1 converges host/hash/IP in kill-chain order, label=0 scatters them."""
    primary_host = host or f"workstation-{seed % 19:02d}"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=seed * 3)
    events: list[IdrEvent] = []

    def add(minutes: int, source: str, severity: str, kind: dict, event_host: str | None = None) -> None:
        events.append(IdrEvent(
            id=str(uuid.UUID(int=(seed * 100 + len(events) + 1) % (1 << 128))),
            timestamp=start + timedelta(minutes=minutes),
            source=source,
            severity=severity,
            kind=kind,
            metadata={"host": event_host or primary_host, "simulation_seed": seed},
        ))

    suspicious_ip = f"203.0.113.{20 + seed % 70}"
    process_id = 7000 + seed
    executable_hash = f"deadbeef{seed:056x}"[-64:]
    host_map = [primary_host] * 6 if label else [
        primary_host,
        f"sensor-{seed % 7}",
        f"workstation-{(seed + 5) % 19:02d}",
        f"server-{seed % 11:02d}",
        f"laptop-{seed % 13:02d}",
        f"storage-{seed % 5:02d}",
    ]
    ip_map = [suspicious_ip] * 5 if label else [
        suspicious_ip,
        f"203.0.113.{(seed + 31) % 90 + 1}",
        f"203.0.113.{(seed + 47) % 90 + 1}",
        f"192.0.2.{(seed + 9) % 90 + 1}",
        f"198.51.100.{(seed + 55) % 90 + 1}",
    ]
    time_map = [8, 10, 12, 17, 19, 23] if label else [19, 8, 23, 10, 17, 12]

    add(time_map[0], "kernel_ebpf", "HIGH", {
        "type": "socket_lineage", "pid": process_id, "tgid": process_id,
        "exe_path": "/tmp/.cache/update", "exe_sha256": executable_hash,
        "dst_ip": ip_map[0], "dst_port": 443, "is_signed": False,
    }, host_map[0])
    add(time_map[1], "sentinel_correlation", "HIGH", {
        "type": "bgp_anomaly",
        "kind": {"kind": "subprefix_hijack_local_infra", "covered_local_prefix": "203.0.113.0/24", "hijacker_asn": 64580},
        "prefix": "203.0.113.0/25", "observed_origin_asn": 64580,
        "legitimate_origin_asn": 64500, "confidence": "high", "dst_ip": ip_map[1],
    }, host_map[1])
    add(time_map[2], "kernel_ebpf", "CRITICAL", {
        "type": "suspicious_beacon", "pid": process_id if label else process_id + 900,
        "exe_path": "/tmp/.cache/update",
        "exe_sha256": executable_hash if label else f"cafebabe{seed:056x}"[-64:],
        "dst_ip": ip_map[2], "asn_owner": "Example Transit",
    }, host_map[2])
    add(time_map[3], "network_zeek", "HIGH", {
        "type": "ntp_time_shift", "offset_seconds": 90.0, "ntp_server": ip_map[3],
    }, host_map[3])
    add(time_map[4], "network_zeek", "CRITICAL", {
        "type": "hsts_time_manipulation", "domain": "update-cdn.example",
        "cert_expiry": "2025-12-01T00:00:00Z", "ntp_shift_seconds": 90.0, "dst_ip": ip_map[4],
    }, host_map[4])
    add(time_map[5], "hardware_nvme", "HIGH", {
        "type": "nvme_latency_anomaly", "device": "nvme0n1", "baseline_us": 120,
        "observed_us": 1900 + seed, "deviation_pct": 1483.0, "concurrent_exfil": True,
    }, host_map[5])
    return sorted(events, key=lambda event: (event.timestamp, event.id))
