"""Tests for engine state serialization."""

from ruler_gepa import RulerAdapter, RulerConfig, RulerEngineState, RulerGEPAEngine


class FakeEvalResult:
    def __init__(self, outputs):
        self.outputs = outputs


class FakeBaseAdapter:
    def evaluate(self, batch, candidate, capture_traces=True):
        return FakeEvalResult(outputs=[candidate["name"]])


def test_export_and_restore_state_round_trip():
    responses = iter(["RANKING: [2, 1]", "RANKING: [2, 1]"])
    adapter = RulerAdapter(
        base_adapter=FakeBaseAdapter(),
        config=RulerConfig(cache_rankings=True),
        judge_fn=lambda prompt: next(responses),
    )
    engine = RulerGEPAEngine(adapter=adapter, rubric="Prefer the new proposal.")
    engine.accept_candidate(
        new_candidate={"name": "new"},
        parent_candidate={"name": "old"},
        minibatch=["ex1", "ex2"],
    )

    state = engine.export_state()
    restored = RulerGEPAEngine.from_state(
        adapter=RulerAdapter(
            base_adapter=FakeBaseAdapter(),
            config=RulerConfig(cache_rankings=True),
            judge_fn=lambda prompt: "RANKING: [1, 2]",
        ),
        state=RulerEngineState.from_dict(state.to_dict()),
        rubric="Prefer the new proposal.",
    )

    assert len(restored.select_frontier()) == 2
    assert restored.adapter.stats["judge_calls"] == 2
    assert restored.adapter.export_ranking_cache()
