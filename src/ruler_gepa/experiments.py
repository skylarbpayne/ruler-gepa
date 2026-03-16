"""Experiment planning helpers for RULER-GEPA."""

from __future__ import annotations

from dataclasses import dataclass

from ruler_gepa.benchmarks import BenchmarkSpec, get_benchmark


@dataclass(frozen=True)
class ExperimentVariant:
    """Represents one method or ablation from the implementation plan."""

    name: str
    description: str


@dataclass(frozen=True)
class PlannedRun:
    """Concrete benchmark × variant experiment run."""

    benchmark: BenchmarkSpec
    variant: ExperimentVariant


BASELINE_VARIANTS: tuple[ExperimentVariant, ...] = (
    ExperimentVariant("GEPA", "Baseline with absolute scoring."),
    ExperimentVariant("RULER-GEPA", "Full relative evaluation and comparative reflection."),
)


ABLATION_VARIANTS: tuple[ExperimentVariant, ...] = (
    ExperimentVariant("Abl-RelativeOnly", "Relative eval with standard reflection."),
    ExperimentVariant("Abl-ReflectionOnly", "Absolute eval with comparative reflection."),
    ExperimentVariant("Abl-NoBradleyTerry", "Win counting instead of BT aggregation."),
    ExperimentVariant("Abl-PairwiseOnly", "Compare only candidate vs parent."),
    ExperimentVariant("Abl-LargerBatch", "Use a larger ranking batch size."),
)


def build_experiment_plan(
    benchmarks: list[str] | None = None,
    include_ablations: bool = True,
) -> list[PlannedRun]:
    """Create the planned experiment matrix from the repo plan."""
    benchmark_names = benchmarks or ["papillon", "ifbench", "hotpotqa"]
    variants = list(BASELINE_VARIANTS)
    if include_ablations:
        variants.extend(ABLATION_VARIANTS)

    runs: list[PlannedRun] = []
    for benchmark_name in benchmark_names:
        benchmark = get_benchmark(benchmark_name)
        for variant in variants:
            runs.append(PlannedRun(benchmark=benchmark, variant=variant))
    return runs
