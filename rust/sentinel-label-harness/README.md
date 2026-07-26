# sentinel-label-harness (machine-local, not in CI)

A red-team **label driver**: it generates synthetic multi-modal windows, runs
**each one through idr-main's real `SentinelCorrelator`**, and labels the window
by the correlator's own verdict — an emitted `ImpossibleState` means the
platform confirmed a campaign (label 1); no emission means it did not (label 0).
The output is `LabeledWindow` NDJSON the engine's `--data` / `validate` path
consumes directly.

The point: the label is the detection platform's **deterministic ground truth**,
not an assertion by the ML engine or by me. This is the honest version of
"drive a kill chain through the sentinel" — it exercises the real correlator,
safely.

## Safety

`IdrConfig::default()` leaves `auto_panic_enabled = false`, so the panic
responder's `execute()` returns immediately — the `ip link set down` and
`nvme format --ses=2` commands are never reached. Windows run through a fresh
in-process correlator; nothing touches the live daemon or its production state
(`/var/lib/idr-sentinel/*`).

## Why machine-local

It path-deps the idr-main working tree (`crates/idr-sentinel`, `crates/idr-common`),
which is the whole correlator + its dependency graph — not available in this
repo's CI. Like `../idr-common-parity`, it has its own `[workspace]` and is
built/run on the machine that has idr-main.

## Run

```sh
cargo run --release -- out.labeled.ndjson 40     # 40 positives + 40 negatives, correlator-labeled
idr-intelligence demo     --data <dir>           # train on it
idr-intelligence validate --data <dir> --data-provenance synthetic-events-correlator-labeled
```

## Honest scope

The EVENTS are synthetic (generated to exercise the correlator's confirmation
recipe: a physics single-hop intercept on a high-trust path + an IGMP→QUIC
correlation, ahead of a multi-modal payload). Only the LABELS are real
(correlator-assigned). So `validate` reports a real verdict but marks it
NON-BINDING — a binding production verdict still needs real captured telemetry.
What this closes is the *label* axis: campaign labels now come from idr-main's
own deterministic correlator instead of a hand-asserted template.
