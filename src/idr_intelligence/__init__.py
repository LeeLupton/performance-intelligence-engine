"""Temporal-graph intelligence for IDR event streams."""

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
