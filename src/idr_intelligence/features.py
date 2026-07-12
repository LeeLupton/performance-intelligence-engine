from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .schema import IdrEvent, KIND_PRIOR, SEVERITY_WEIGHT

FEATURE_NAMES = [
    "severity",
    "kind_prior",
    "delta_seconds_log",
    "has_process",
    "unsigned_process",
    "has_ip",
    "has_hash",
    "has_domain",
    "has_asn_or_prefix",
    "has_hardware",
    "concurrent_exfil",
    "production_bgp_anomaly",
    "source_kernel",
    "source_network",
    "source_hardware",
    "source_correlation",
    "kind_hash_0",
    "kind_hash_1",
    "kind_hash_2",
    "kind_hash_3",
]
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class EventProjection:
    event: IdrEvent
    entities: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    features: np.ndarray


def project_event(event: IdrEvent, delta_seconds: float = 0.0) -> EventProjection:
    entities = extract_entities(event)
    edges = tuple(_derive_edges(event, entities))
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    kind = event.kind
    features[0] = SEVERITY_WEIGHT.get(event.severity, 0.15)
    features[1] = KIND_PRIOR.get(event.kind_type, 0.18)
    features[2] = np.log1p(max(delta_seconds, 0.0)) / 12.0
    features[3] = float("pid" in kind or "tgid" in kind)
    features[4] = float(kind.get("is_signed") is False)
    features[5] = float(any(key.endswith("_ip") or key == "dest_ips" for key in kind))
    features[6] = float("exe_sha256" in kind or bool(kind.get("sha256")))
    features[7] = float(any(key in kind for key in ("domain", "sni", "ptr_query")))
    features[8] = float(any(key in kind for key in ("prefix", "observed_origin_asn", "asn_owner")))
    features[9] = float(event.kind_type in {"nvme_latency_anomaly", "mac_flapping", "rtc_clock_divergence"})
    features[10] = float(bool(kind.get("concurrent_exfil")))
    features[11] = float(_is_production_bgp_anomaly(kind))
    source_group = _source_group(event.source)
    features[12 + source_group] = 1.0
    digest = hashlib.blake2s(event.kind_type.encode(), digest_size=1).digest()[0]
    features[16 + digest % 4] = 1.0
    return EventProjection(event=event, entities=entities, edges=edges, features=features)


def extract_entities(event: IdrEvent) -> tuple[str, ...]:
    kind = event.kind
    host = str(event.metadata.get("host") or event.metadata.get("hostname") or "unknown-host")
    entities = [f"host:{host}"]
    pid = kind.get("pid") or kind.get("tgid")
    if pid is not None:
        entities.append(f"process:{host}:{pid}")
    for key in ("exe_sha256", "sha256"):
        if kind.get(key):
            entities.append(f"hash:{str(kind[key]).lower()}")
    for key in ("src_ip", "dst_ip", "forward_ip", "reversed_ip", "ntp_server", "gateway_ip"):
        if kind.get(key):
            entities.append(f"ip:{kind[key]}")
    for value in kind.get("dest_ips", []) or []:
        entities.append(f"ip:{value}")
    if kind.get("prefix"):
        entities.append(f"prefix:{kind['prefix']}")
    for key in ("observed_origin_asn", "legitimate_origin_asn"):
        if kind.get(key) is not None:
            entities.append(f"asn:{kind[key]}")
    for key in ("domain", "ptr_query"):
        if kind.get(key):
            entities.append(f"domain:{str(kind[key]).lower().rstrip('.')}")
    if kind.get("device"):
        entities.append(f"device:{host}:{kind['device']}")
    return tuple(dict.fromkeys(entities))


def _derive_edges(event: IdrEvent, entities: tuple[str, ...]):
    host = next((x for x in entities if x.startswith("host:")), None)
    process = next((x for x in entities if x.startswith("process:")), None)
    hashes = [x for x in entities if x.startswith("hash:")]
    ips = [x for x in entities if x.startswith("ip:")]
    prefixes = [x for x in entities if x.startswith("prefix:")]
    asns = [x for x in entities if x.startswith("asn:")]
    domains = [x for x in entities if x.startswith("domain:")]
    devices = [x for x in entities if x.startswith("device:")]
    if host and process:
        yield host, process, "executes"
    for digest in hashes:
        yield process or host, digest, "identified_by"
    for ip in ips:
        yield process or host, ip, "connects_to"
    for prefix in prefixes:
        for ip in ips:
            yield ip, prefix, "belongs_to"
        for asn in asns:
            yield prefix, asn, "originated_by"
    for domain in domains:
        for ip in ips:
            yield domain, ip, "resolves_or_connects"
        if host:
            yield host, domain, "queries_or_visits"
    for device in devices:
        if host:
            yield host, device, "contains"


def _is_production_bgp_anomaly(kind: dict) -> bool:
    if kind.get("type") != "bgp_anomaly":
        return False
    nested = kind.get("kind")
    return isinstance(nested, dict) and nested.get("kind") == "subprefix_hijack_local_infra"


def _source_group(source: str) -> int:
    if source == "kernel_ebpf":
        return 0
    if source in {"network_zeek", "network_suricata"}:
        return 1
    if source.startswith("hardware_"):
        return 2
    return 3
