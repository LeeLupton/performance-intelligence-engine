"""Temporal-graph intelligence for IDR event streams."""

import logging as _logging

from .campaigns import CampaignRegistry
from .config import ENGINE_VERSION
from .export import OnnxStreamScorer, export_streaming_bundle
from .pipeline import IntelligenceFinding, score_events
from .schema import IdrEvent
from .streaming import StreamingScorer

__all__ = [
    "CampaignRegistry",
    "IdrEvent",
    "IntelligenceFinding",
    "OnnxStreamScorer",
    "StreamingScorer",
    "export_streaming_bundle",
    "score_events",
]
__version__ = ENGINE_VERSION

# Silent by default: importing the library emits nothing and scoring stays
# byte-for-byte deterministic on stdout. The CLI opts into stderr logging via
# observability.configure_logging().
_logging.getLogger("idr_intelligence").addHandler(_logging.NullHandler())
