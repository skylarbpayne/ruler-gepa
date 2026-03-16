"""Tests for benchmark and experiment scaffolding."""

from ruler_gepa import build_experiment_plan, get_benchmark, list_benchmarks


def test_benchmark_registry_is_prioritized():
    benchmarks = list_benchmarks()

    assert benchmarks[0].name == "PAPILLON"
    assert get_benchmark("ifbench").task_type == "instruction following"


def test_experiment_plan_includes_baselines_and_ablations():
    runs = build_experiment_plan(["papillon"], include_ablations=True)

    variant_names = {run.variant.name for run in runs}
    assert "GEPA" in variant_names
    assert "RULER-GEPA" in variant_names
    assert "Abl-RelativeOnly" in variant_names
    assert len(runs) == 7
