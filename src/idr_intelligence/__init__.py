"""Temporal-graph intelligence for IDR event streams."""

from .pipeline import IntelligenceFinding, score_events
from .schema import IdrEvent

__all__ = ["IdrEvent", "IntelligenceFinding", "score_events"]
__version__ = "0.1.0"
