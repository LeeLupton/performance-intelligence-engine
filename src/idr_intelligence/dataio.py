"""Loading labeled real-campaign windows from *.labeled.ndjson files.

Each line of a .labeled.ndjson file is one serialized LabeledWindow:
{"window_id": ..., "label": 0|1, "events": [IdrEvent, ...]}. A directory is
scanned for every *.labeled.ndjson inside it; a file path loads just itself.
Errors carry file and line context so a bad export is findable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import LabeledWindow


def load_labeled_windows(path: str | Path) -> list[LabeledWindow]:
    """Load every labeled window under path, sorted by window start time."""
    root = Path(path)
    if root.is_dir():
        files = sorted(root.glob("*.labeled.ndjson"))
        if not files:
            raise ValueError(f"no *.labeled.ndjson files under {root}")
    elif root.is_file():
        files = [root]
    else:
        raise ValueError(f"no such file or directory: {root}")
    windows: list[LabeledWindow] = []
    for file in files:
        for line_number, line in enumerate(file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                windows.append(LabeledWindow.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{file}:{line_number}: invalid labeled window: {exc}") from exc
    return sorted(windows, key=lambda window: (window.start, window.window_id))
