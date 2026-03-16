"""
RULER-GEPA: Relative evaluation for prompt optimization.

Combines GEPA's evolutionary prompt optimization with RULER's
relative LLM-as-judge evaluation strategy.
"""

from ruler_gepa.adapter import RelativeEvaluationResult, RulerAdapter
from ruler_gepa.aggregation import BradleyTerryAggregator, CopelandAggregator, EloAggregator
from ruler_gepa.benchmarks import BenchmarkSpec, get_benchmark, list_benchmarks
from ruler_gepa.config import RulerConfig
from ruler_gepa.engine import RulerGEPAEngine
from ruler_gepa.experiments import (
    ABLATION_VARIANTS,
    BASELINE_VARIANTS,
    ExperimentVariant,
    PlannedRun,
    build_experiment_plan,
)
from ruler_gepa.proposer import (
    build_comparative_reflective_dataset,
    build_reflection_payload,
    summarize_candidate,
)
from ruler_gepa.state import CandidateRecord, EngineStats, RulerEngineState

__version__ = "0.1.0"
__all__ = [
    "BradleyTerryAggregator",
    "BenchmarkSpec",
    "CandidateRecord",
    "CopelandAggregator",
    "EloAggregator",
    "EngineStats",
    "ExperimentVariant",
    "BASELINE_VARIANTS",
    "ABLATION_VARIANTS",
    "PlannedRun",
    "RelativeEvaluationResult",
    "RulerAdapter",
    "RulerConfig",
    "RulerGEPAEngine",
    "build_experiment_plan",
    "build_comparative_reflective_dataset",
    "build_reflection_payload",
    "RulerEngineState",
    "get_benchmark",
    "list_benchmarks",
    "summarize_candidate",
]
