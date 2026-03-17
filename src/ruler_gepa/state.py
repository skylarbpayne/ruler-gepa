"""State objects for the experimental RULER-GEPA engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    """Tracked state for a candidate in the frontier."""

    candidate_id: str
    candidate: dict[str, str]
    parent_id: str | None = None
    score: float = 0.0
    accepted: bool = False
    wins: int = 0
    comparisons: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStats:
    """Operational counters for relative evaluation runs."""

    judge_calls: int = 0
    cache_hits: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    comparisons: int = 0


@dataclass
class RulerEngineState:
    """Serializable engine state."""

    records: list[CandidateRecord] = field(default_factory=list)
    stats: EngineStats = field(default_factory=EngineStats)
    ranking_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-safe dictionary."""
        return {
            "records": [asdict(record) for record in self.records],
            "stats": asdict(self.stats),
            "ranking_cache": self.ranking_cache,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RulerEngineState:
        """Reconstruct state from serialized data."""
        return cls(
            records=[CandidateRecord(**record) for record in payload.get("records", [])],
            stats=EngineStats(**payload.get("stats", {})),
            ranking_cache=payload.get("ranking_cache", {}),
        )
