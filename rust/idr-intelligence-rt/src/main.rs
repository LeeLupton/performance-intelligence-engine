//! CLI mirroring `idr-intelligence stream`: NDJSON events in, one finding out.

use std::path::PathBuf;
use std::process::ExitCode;

use idr_intelligence_rt::event::read_ndjson;
use idr_intelligence_rt::scorer::StreamingScorer;

const USAGE: &str = "usage: idr-intelligence-rt --model-dir DIR [EVENTS|-] [--max-nodes N] [--suppress RULE]... [--top-k K]

Scores newline-delimited IdrEvent JSON (a file, or stdin when EVENTS is '-' or
omitted) one event at a time over the exported step cell, then emits one
finding JSON with an `evictions` audit list — the same output shape as
`idr-intelligence stream`. Events are sorted by (timestamp, id) before ingest.

--max-nodes defaults to 4096 (mirroring `idr-intelligence stream`) so the
dense N x N scoring adjacency stays bounded on untrusted streams; pass
--max-nodes 0 to disable the bound.";

/// Default entity budget — keeps finding()'s dense adjacency bounded even
/// when the stream is untrusted. Matches the Python stream CLI default.
const DEFAULT_MAX_NODES: usize = 4096;

struct Args {
    model_dir: PathBuf,
    events: Option<PathBuf>,
    max_nodes: Option<usize>,
    suppress: Vec<String>,
    top_k: Option<usize>,
}

fn parse_args(argv: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut model_dir = None;
    let mut events = None;
    let mut max_nodes = Some(DEFAULT_MAX_NODES);
    let mut suppress = Vec::new();
    let mut top_k = None;
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
    })
}

fn run() -> Result<(), String> {
    let args = parse_args(std::env::args().skip(1))?;
    let text = match &args.events {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
        None => std::io::read_to_string(std::io::stdin())
            .map_err(|error| format!("cannot read stdin: {error}"))?,
    };
    let mut events = read_ndjson(&text)?;
    events.sort_by(|a, b| (a.timestamp, &a.id).cmp(&(b.timestamp, &b.id)));
    let mut scorer = StreamingScorer::open(&args.model_dir, args.max_nodes)?;
    for event in &events {
        scorer.ingest(event)?;
    }
    let finding = scorer.finding(args.top_k, &args.suppress)?;
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
