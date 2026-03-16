"""Minimal engine for RULER-style prompt optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ruler_gepa.adapter import RulerAdapter
from ruler_gepa.proposer import build_reflection_payload
from ruler_gepa.state import CandidateRecord, EngineStats, RulerEngineState


@dataclass
class AcceptanceDecision:
    """Result of comparing a proposal against its parent."""

    accepted: bool
    win_rate: float
    wins: int
    total: int
    comparisons: list[dict[str, Any]] = field(default_factory=list)


class RulerGEPAEngine:
    """Experimental engine implementing the plan's relative acceptance loop."""

    def __init__(
        self,
        adapter: RulerAdapter,
        trainset: list[Any] | None = None,
        valset: list[Any] | None = None,
        rubric: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.trainset = list(trainset or [])
        self.valset = list(valset or [])
        self.rubric = rubric
        self.frontier: list[CandidateRecord] = []

    def register_candidate(
        self,
        candidate: dict[str, str],
        *,
        parent_id: str | None = None,
        accepted: bool = False,
        wins: int = 0,
        comparisons: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateRecord:
        """Add or update a candidate record in the frontier."""
        candidate_id = self.adapter._hash_candidate(candidate)
        record = CandidateRecord(
            candidate_id=candidate_id,
            candidate=candidate,
            parent_id=parent_id,
            score=self.adapter.get_candidate_score(candidate),
            accepted=accepted,
            wins=wins,
            comparisons=comparisons,
            metadata=dict(metadata or {}),
        )
        self.frontier = [existing for existing in self.frontier if existing.candidate_id != candidate_id]
        self.frontier.append(record)
        self.frontier.sort(key=lambda item: item.score, reverse=True)
        return record

    def accept_candidate(
        self,
        new_candidate: dict[str, str],
        parent_candidate: dict[str, str],
        minibatch: list[Any],
    ) -> AcceptanceDecision:
        """Accept if the proposal wins on a strict majority of examples."""
        accepted, win_rate, comparisons = self.adapter.compare_to_parent(
            new_candidate=new_candidate,
            parent_candidate=parent_candidate,
            minibatch=minibatch,
            rubric=self.rubric,
        )
        wins = sum(1 for comparison in comparisons if comparison["new_wins"])
        decision = AcceptanceDecision(
            accepted=accepted,
            win_rate=win_rate,
            wins=wins,
            total=len(comparisons),
            comparisons=comparisons,
        )
        self.register_candidate(
            parent_candidate,
            accepted=False,
            metadata={"role": "parent"},
        )
        self.register_candidate(
            new_candidate,
            parent_id=self.adapter._hash_candidate(parent_candidate),
            accepted=accepted,
            wins=wins,
            comparisons=len(comparisons),
            metadata={
                "win_rate": win_rate,
                "reflection": build_reflection_payload(new_candidate, comparisons),
            },
        )
        return decision

    def select_frontier(self, limit: int | None = None) -> list[CandidateRecord]:
        """Return the current frontier ordered by aggregated score."""
        frontier = sorted(self.frontier, key=lambda item: item.score, reverse=True)
        if limit is None:
            return frontier
        return frontier[:limit]

    def score_candidates(self, candidates: list[dict[str, str]]) -> dict[str, float]:
        """Return aggregated scores for a set of candidates."""
        return {
            self.adapter._hash_candidate(candidate): self.adapter.get_candidate_score(candidate)
            for candidate in candidates
        }

    def export_state(self) -> RulerEngineState:
        """Serialize the current engine state."""
        stats = EngineStats(**self.adapter.stats)
        return RulerEngineState(
            records=list(self.frontier),
            stats=stats,
            ranking_cache=self.adapter.export_ranking_cache(),
        )

    @classmethod
    def from_state(
        cls,
        adapter: RulerAdapter,
        state: RulerEngineState,
        trainset: list[Any] | None = None,
        valset: list[Any] | None = None,
        rubric: str | None = None,
    ) -> RulerGEPAEngine:
        """Restore an engine from serialized state."""
        engine = cls(adapter=adapter, trainset=trainset, valset=valset, rubric=rubric)
        engine.frontier = list(state.records)
        adapter.import_ranking_cache(state.ranking_cache)
        adapter.stats.update(
            {
                "judge_calls": state.stats.judge_calls,
                "cache_hits": state.stats.cache_hits,
                "prompt_chars": state.stats.prompt_chars,
                "completion_chars": state.stats.completion_chars,
                "comparisons": state.stats.comparisons,
            }
        )
        return engine
