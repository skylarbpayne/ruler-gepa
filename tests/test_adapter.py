"""Tests for the RULER adapter."""

from dataclasses import dataclass

from ruler_gepa import RelativeEvaluationResult, RulerAdapter, RulerConfig


@dataclass
class FakeEvalResult:
    outputs: list[str]


class FakeBaseAdapter:
    def evaluate(self, batch, candidate, capture_traces=True):
        example = batch[0]
        suffix = candidate.get("suffix", "")
        return FakeEvalResult(outputs=[f"{example['prompt']}::{suffix}"])


def test_relative_evaluate_returns_ranking_and_metadata():
    adapter = RulerAdapter(
        base_adapter=FakeBaseAdapter(),
        config=RulerConfig(cache_rankings=False, deduplicate_common_prefix=False),
        judge_fn=lambda prompt: "RANKING: [2, 1]",
    )

    ranking, metadata = adapter.relative_evaluate(
        candidates=[{"suffix": "parent"}, {"suffix": "child"}],
        example={"prompt": "task"},
        rubric="Prefer the second answer.",
    )

    assert ranking == [1, 0]
    assert metadata["outputs"] == ["task::parent", "task::child"]
    assert metadata["cached"] is False


def test_ranking_cache_short_circuits_second_judge_call():
    calls = {"count": 0}

    def judge_fn(prompt: str) -> str:
        calls["count"] += 1
        return "RANKING: [1, 2]"

    adapter = RulerAdapter(
        base_adapter=FakeBaseAdapter(),
        config=RulerConfig(cache_rankings=True),
        judge_fn=judge_fn,
    )
    candidates = [{"suffix": "a"}, {"suffix": "b"}]
    example = {"prompt": "cached"}

    first = adapter.evaluate_and_record(candidates, example)
    second = adapter.evaluate_and_record(candidates, example)

    assert isinstance(first, RelativeEvaluationResult)
    assert first.ranking == [0, 1]
    assert second.metadata["cached"] is True
    assert calls["count"] == 1


def test_compare_to_parent_uses_strict_majority():
    responses = iter(["RANKING: [2, 1]", "RANKING: [1, 2]", "RANKING: [2, 1]"])

    adapter = RulerAdapter(
        base_adapter=FakeBaseAdapter(),
        config=RulerConfig(cache_rankings=False, min_win_rate=0.5),
        judge_fn=lambda prompt: next(responses),
    )

    accepted, win_rate, comparisons = adapter.compare_to_parent(
        new_candidate={"suffix": "new"},
        parent_candidate={"suffix": "old"},
        minibatch=[{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}],
    )

    assert accepted is True
    assert win_rate == 2 / 3
    assert len(comparisons) == 3
