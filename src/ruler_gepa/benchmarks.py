"""Benchmark registry for planned RULER-GEPA experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata for a benchmark in the experiment plan."""

    name: str
    task_type: str
    train_examples: int
    val_examples: int
    good_for_ruler: bool
    rationale: str
    priority: int


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "papillon": BenchmarkSpec(
        name="PAPILLON",
        task_type="structured extraction",
        train_examples=326,
        val_examples=326,
        good_for_ruler=True,
        rationale="Already uses LLM-as-judge style evaluation.",
        priority=1,
    ),
    "ifbench": BenchmarkSpec(
        name="IFBench",
        task_type="instruction following",
        train_examples=200,
        val_examples=200,
        good_for_ruler=True,
        rationale="Multi-aspect evaluation makes relative ranking useful.",
        priority=2,
    ),
    "hotpotqa": BenchmarkSpec(
        name="HotPotQA",
        task_type="multi-hop QA",
        train_examples=500,
        val_examples=500,
        good_for_ruler=True,
        rationale="Good fit for comparing reasoning quality across candidates.",
        priority=3,
    ),
}


def get_benchmark(name: str) -> BenchmarkSpec:
    """Fetch a benchmark definition by normalized name."""
    key = name.lower()
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark: {name}")
    return BENCHMARKS[key]


def list_benchmarks() -> list[BenchmarkSpec]:
    """Return benchmarks in the recommended execution order."""
    return sorted(BENCHMARKS.values(), key=lambda spec: spec.priority)
