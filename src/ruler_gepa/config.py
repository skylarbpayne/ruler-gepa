"""Configuration for RULER-GEPA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RulerConfig:
    """Configuration for RULER-style relative evaluation.

    Attributes:
        enabled: Whether to use RULER-style evaluation.
        judge_lm: Model to use for ranking judgments.
        comparison_batch_size: Number of candidates to rank simultaneously (2-8).
        rubric: Custom ranking rubric. If None, uses objective description.
        use_objective_as_rubric: Whether to derive rubric from optimization objective.
        aggregation: Method to aggregate rankings into scores.
        bt_learning_rate: Learning rate for Bradley-Terry updates.
        min_win_rate: Minimum win rate against parent to accept candidate.
        deduplicate_common_prefix: RULER optimization - dedupe shared candidate parts.
        cache_rankings: Cache rankings to avoid re-judging same comparisons.
    """

    # Core settings
    enabled: bool = True
    judge_lm: str = "openai/gpt-4.1"
    comparison_batch_size: int = 4

    # Rubric configuration
    rubric: str | None = None
    use_objective_as_rubric: bool = True

    # Ranking aggregation
    aggregation: Literal["bradley-terry", "elo", "copeland"] = "bradley-terry"
    bt_learning_rate: float = 0.1
    elo_k_factor: float = 32.0

    # Acceptance criteria
    min_win_rate: float = 0.5

    # Efficiency optimizations
    deduplicate_common_prefix: bool = True
    cache_rankings: bool = True

    # Advanced
    include_frontier_in_comparison: bool = True
    max_frontier_samples: int = 2

    def __post_init__(self):
        if self.comparison_batch_size < 2:
            raise ValueError("comparison_batch_size must be at least 2")
        if self.comparison_batch_size > 8:
            raise ValueError("comparison_batch_size > 8 may degrade judge quality")
        if not 0 < self.min_win_rate <= 1:
            raise ValueError("min_win_rate must be in (0, 1]")
        if self.aggregation == "bradley-terry" and self.bt_learning_rate <= 0:
            raise ValueError("bt_learning_rate must be > 0")
        if self.aggregation == "elo" and self.elo_k_factor <= 0:
            raise ValueError("elo_k_factor must be > 0")
