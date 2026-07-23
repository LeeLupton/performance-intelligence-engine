"""Temporal-graph intelligence for IDR event streams."""

from .campaigns import CampaignRegistry
from .config import ENGINE_VERSION
from .pipeline import IntelligenceFinding, score_events
from .schema import IdrEvent
from .streaming import StreamingScorer

__all__ = ["CampaignRegistry", "IdrEvent", "IntelligenceFinding", "StreamingScorer", "score_events"]
__version__ = ENGINE_VERSION
