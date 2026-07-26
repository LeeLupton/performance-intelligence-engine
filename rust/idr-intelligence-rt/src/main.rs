//! CLI mirroring `idr-intelligence stream`: NDJSON events in, one finding out.

use std::path::PathBuf;
use std::process::ExitCode;

use idr_intelligence_rt::event::read_ndjson;
use idr_intelligence_rt::log::{Level, Logger};
use idr_intelligence_rt::scorer::StreamingScorer;
use serde_json::json;

const USAGE: &str = "usage: idr-intelligence-rt --model-dir DIR [EVENTS|-] [--max-nodes N] [--suppress RULE]... [--top-k K]

Scores newline-delimited IdrEvent JSON (a file, or stdin when EVENTS is '-' or
omitted) one event at a time over the exported step cell, then emits one
finding JSON with an `evictions` audit list — the same output shape as
`idr-intelligence stream`. Events are sorted by (timestamp, id) before ingest.

--max-nodes defaults to 4096 (mirroring `idr-intelligence stream`) so the
dense N x N scoring adjacency stays bounded on untrusted streams; pass
--max-nodes 0 to disable the bound. --log-level (or IDR_RT_LOG) emits
JSON-lines operational logs to stderr; the stdout finding is unchanged.";

/// Default entity budget — keeps finding()'s dense adjacency bounded even
/// when the stream is untrusted. Matches the Python stream CLI default.
const DEFAULT_MAX_NODES: usize = 4096;

struct Args {
    model_dir: PathBuf,
    events: Option<PathBuf>,
    max_nodes: Option<usize>,
    suppress: Vec<String>,
    top_k: Option<usize>,
    log_level: Option<String>,
}

fn parse_args(argv: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut model_dir = None;
    let mut events = None;
    let mut max_nodes = Some(DEFAULT_MAX_NODES);
    let mut suppress = Vec::new();
    let mut top_k = None;
    let mut log_level = None;
    let mut argv = argv.into_iter();
    while let Some(argument) = argv.next() {
        let mut value_for = |flag: &str| argv.next().ok_or(format!("{flag} requires a value"));
        match argument.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value_for("--model-dir")?)),
            "--max-nodes" => {
                let bound = value_for("--max-nodes")?
                    .parse::<usize>()
                    .map_err(|error| format!("--max-nodes: {error}"))?;
                max_nodes = if bound == 0 { None } else { Some(bound) };
            }
            "--suppress" => suppress.push(value_for("--suppress")?),
            "--log-level" => log_level = Some(value_for("--log-level")?),
            "--top-k" => {
                top_k = Some(
                    value_for("--top-k")?
                        .parse::<usize>()
                        .map_err(|error| format!("--top-k: {error}"))?,
                )
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            "-" => events = None,
            other if !other.starts_with("--") && events.is_none() => {
                events = Some(PathBuf::from(other))
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }
    let model_dir = model_dir.ok_or(format!("--model-dir is required\n\n{USAGE}"))?;
    Ok(Args {
        model_dir,
        events,
        max_nodes,
        suppress,
        top_k,
        log_level,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args(std::env::args().skip(1))?;
    let logger = Logger::from_flag(args.log_level.as_deref())?;
    let source = args
        .events
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "stdin".to_string());
    let text = match &args.events {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
        None => std::io::read_to_string(std::io::stdin())
            .map_err(|error| format!("cannot read stdin: {error}"))?,
    };
    let mut events = read_ndjson(&text)?;
    events.sort_by(|a, b| (a.timestamp, &a.id).cmp(&(b.timestamp, &b.id)));
    logger.event(
        Level::Info,
        "events_read",
        json!({"events": events.len(), "source": source}),
    );
    let mut scorer =
        StreamingScorer::open(&args.model_dir, args.max_nodes).inspect_err(|error| {
            logger.event(
                Level::Error,
                "model_load_failed",
                json!({"model_dir": args.model_dir.display().to_string(), "reason": error}),
            );
        })?;
    logger.event(
        Level::Info,
        "model_loaded",
        json!({
            "model_dir": args.model_dir.display().to_string(),
            "feature_schema_hash": scorer.manifest.feature_schema_hash,
            "feature_dim": scorer.manifest.model.feature_dim,
            "time_mode": scorer.manifest.model.time_mode,
            "calibration": scorer.manifest.calibration.label,
            "max_nodes": args.max_nodes,
        }),
    );
    let started = std::time::Instant::now();
    for event in &events {
        scorer.ingest(event)?;
    }
    let finding = scorer.finding(args.top_k, &args.suppress)?;
    let elapsed_ms = (started.elapsed().as_secs_f64() * 1000.0 * 1000.0).round() / 1000.0;
    logger.event(
        Level::Info,
        "finding_scored",
        json!({
            "campaign_id": finding.campaign_id,
            "escalation_probability": finding.escalation_probability,
            "graph_nodes": finding.graph_nodes,
            "events": scorer.events_seen,
            "evictions": scorer.evictions.len(),
            "elapsed_ms": elapsed_ms,
        }),
    );
    let mut payload = serde_json::to_value(&finding).map_err(|error| error.to_string())?;
    payload["evictions"] =
        serde_json::to_value(&scorer.evictions).map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_MAX_NODES, parse_args};

    fn args(list: &[&str]) -> super::Args {
        parse_args(list.iter().map(|s| s.to_string())).unwrap()
    }

    #[test]
    fn max_nodes_is_bounded_by_default_and_zero_disables() {
        assert_eq!(
            args(&["--model-dir", "m"]).max_nodes,
            Some(DEFAULT_MAX_NODES)
        );
        assert_eq!(
            args(&["--model-dir", "m", "--max-nodes", "64"]).max_nodes,
            Some(64)
        );
        assert_eq!(
            args(&["--model-dir", "m", "--max-nodes", "0"]).max_nodes,
            None
        );
    }

    #[test]
    fn model_dir_is_required() {
        assert!(parse_args(["--max-nodes".to_string(), "5".to_string()]).is_err());
    }
}
