"""Tests for the experimental engine."""

from dataclasses import dataclass

from ruler_gepa import RulerAdapter, RulerConfig, RulerGEPAEngine


@dataclass
class FakeEvalResult:
    outputs: list[str]


class FakeBaseAdapter:
    def evaluate(self, batch, candidate, capture_traces=True):
        return FakeEvalResult(outputs=[candidate["name"]])


def test_accept_candidate_reports_counts():
    responses = iter(["RANKING: [2, 1]", "RANKING: [2, 1]", "RANKING: [1, 2]"])
    adapter = RulerAdapter(
        base_adapter=FakeBaseAdapter(),
        config=RulerConfig(cache_rankings=False),
        judge_fn=lambda prompt: next(responses),
    )
    engine = RulerGEPAEngine(adapter=adapter, rubric="Prefer the new proposal.")

    decision = engine.accept_candidate(
        new_candidate={"name": "new"},
        parent_candidate={"name": "old"},
        minibatch=["ex1", "ex2", "ex3"],
    )

    assert decision.accepted is True
    assert decision.wins == 2
    assert decision.total == 3
    assert decision.win_rate == 2 / 3


def test_score_candidates_returns_hashed_scores():
    adapter = RulerAdapter(
        base_adapter=FakeBaseAdapter(),
        config=RulerConfig(cache_rankings=False),
        judge_fn=lambda prompt: "RANKING: [1, 2]",
    )
    engine = RulerGEPAEngine(adapter=adapter)
    candidates = [{"name": "a"}, {"name": "b"}]

    adapter.relative_evaluate(candidates=candidates, example="example")
    scores = engine.score_candidates(candidates)

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores.values())
