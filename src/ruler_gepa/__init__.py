"""
RULER-GEPA: Relative evaluation for prompt optimization.

Combines GEPA's evolutionary prompt optimization with RULER's
relative LLM-as-judge evaluation strategy.
"""

from ruler_gepa.config import RulerConfig
from ruler_gepa.adapter import RulerAdapter
from ruler_gepa.aggregation import BradleyTerryAggregator

__version__ = "0.1.0"
__all__ = ["RulerConfig", "RulerAdapter", "BradleyTerryAggregator"]
