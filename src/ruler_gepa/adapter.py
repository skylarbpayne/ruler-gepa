"""RULER adapter for GEPA.

Wraps any GEPAAdapter with RULER-style relative evaluation.
"""

from typing import Any, Dict, List, Tuple
import hashlib
import json
import re

from ruler_gepa.config import RulerConfig
from ruler_gepa.aggregation import create_aggregator, BradleyTerryAggregator
from ruler_gepa.prompts import build_ranking_prompt


class RulerAdapter:
    """Wraps a GEPA adapter with RULER-style relative evaluation.
    
    Instead of returning absolute scores, uses an LLM-as-judge to rank
    multiple candidates on each example, then converts rankings to
    Bradley-Terry (or other) scores.
    
    The base adapter is still used for:
    - Trace capture (needed for reflection)
    - Running candidates to get outputs
    
    This adapter adds:
    - relative_evaluate(): Rank multiple candidates on single example
    - Score aggregation via Bradley-Terry model
    - Comparative feedback for enhanced reflection
    
    Example:
        ```python
        base = DefaultAdapter(...)
        ruler = RulerAdapter(base, config=RulerConfig())
        
        # Rank 4 candidates on one example
        ranking = ruler.relative_evaluate(
            candidates=[cand_a, cand_b, cand_c, cand_d],
            example={"input": "..."},
        )
        # ranking = [2, 0, 3, 1] means cand_c > cand_a > cand_d > cand_b
        ```
    """
    
    def __init__(
        self,
        base_adapter: Any,  # GEPAAdapter
        config: RulerConfig | None = None,
        judge_fn: Any | None = None,  # Optional custom judge function
    ):
        self.base = base_adapter
        self.config = config or RulerConfig()
        self.judge_fn = judge_fn
        
        # Create score aggregator
        self.aggregator = create_aggregator(
            self.config.aggregation,
            learning_rate=self.config.bt_learning_rate,
        )
        
        # Cache for rankings (hash of comparison -> ranking)
        self._ranking_cache: Dict[str, List[int]] = {}
    
    def evaluate(
        self,
        batch: List[Any],
        candidate: Dict[str, str],
        capture_traces: bool = True,
    ) -> Any:
        """Pass through to base adapter for standard evaluation.
        
        This is used for trace capture during reflection.
        For acceptance decisions, use relative_evaluate() instead.
        """
        return self.base.evaluate(batch, candidate, capture_traces)
    
    def relative_evaluate(
        self,
        candidates: List[Dict[str, str]],
        example: Any,
        rubric: str | None = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Rank candidates on a single example using LLM-as-judge.
        
        Args:
            candidates: List of candidate configurations to compare.
            example: Single example to evaluate on.
            rubric: Evaluation criteria. If None, uses config.rubric.
        
        Returns:
            Tuple of:
            - ranking: Indices of candidates from best to worst.
                       e.g., [2, 0, 1] means candidates[2] is best.
            - metadata: Dict with outputs, judge_response, etc.
        """
        n = len(candidates)
        if n < 2:
            raise ValueError("Need at least 2 candidates to compare")
        
        # Check cache
        cache_key = self._cache_key(candidates, example)
        if self.config.cache_rankings and cache_key in self._ranking_cache:
            return self._ranking_cache[cache_key], {"cached": True}
        
        # 1. Run each candidate to get outputs
        outputs = []
        for candidate in candidates:
            result = self.base.evaluate([example], candidate, capture_traces=False)
            output = result.outputs[0] if result.outputs else ""
            outputs.append(output)
        
        # 2. Build ranking prompt
        rubric = rubric or self.config.rubric or "Rank by overall quality and correctness."
        input_text = self._format_example(example)
        
        prompt = build_ranking_prompt(
            input_text=input_text,
            candidates=candidates,
            outputs=outputs,
            rubric=rubric,
            deduplicate=self.config.deduplicate_common_prefix,
        )
        
        # 3. Call judge
        if self.judge_fn:
            response = self.judge_fn(prompt)
        else:
            response = self._call_judge(prompt)
        
        # 4. Parse ranking
        ranking = self._parse_ranking(response, n)
        
        # 5. Update aggregator
        candidate_ids = [self._hash_candidate(c) for c in candidates]
        self.aggregator.update_from_ranking(candidate_ids, ranking)
        
        # 6. Cache result
        if self.config.cache_rankings:
            self._ranking_cache[cache_key] = ranking
        
        metadata = {
            "outputs": outputs,
            "judge_response": response,
            "candidate_ids": candidate_ids,
            "cached": False,
        }
        
        return ranking, metadata
    
    def get_candidate_score(self, candidate: Dict[str, str]) -> float:
        """Get aggregated score for a candidate."""
        candidate_id = self._hash_candidate(candidate)
        return self.aggregator.get_score(candidate_id)
    
    def compare_to_parent(
        self,
        new_candidate: Dict[str, str],
        parent_candidate: Dict[str, str],
        minibatch: List[Any],
        rubric: str | None = None,
    ) -> Tuple[bool, float, List[Dict]]:
        """Compare new candidate against parent on minibatch.
        
        Args:
            new_candidate: Proposed new candidate.
            parent_candidate: Current parent candidate.
            minibatch: Examples to evaluate on.
            rubric: Evaluation criteria.
        
        Returns:
            Tuple of:
            - accepted: Whether new_candidate should be accepted.
            - win_rate: Fraction of examples where new beats parent.
            - comparisons: Detailed comparison results for reflection.
        """
        wins = 0
        comparisons = []
        
        for example in minibatch:
            candidates = [parent_candidate, new_candidate]
            ranking, metadata = self.relative_evaluate(candidates, example, rubric)
            
            # new_candidate is index 1
            new_rank = ranking.index(1)
            parent_rank = ranking.index(0)
            new_wins = new_rank < parent_rank
            
            if new_wins:
                wins += 1
            
            comparisons.append({
                "example": example,
                "ranking": ranking,
                "new_wins": new_wins,
                "outputs": metadata["outputs"],
                "candidates": candidates,
            })
        
        win_rate = wins / len(minibatch)
        accepted = win_rate > self.config.min_win_rate
        
        return accepted, win_rate, comparisons
    
    def _call_judge(self, prompt: str) -> str:
        """Call LLM judge to get ranking."""
        try:
            import litellm
            
            response = litellm.completion(
                model=self.config.judge_lm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for consistency
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("litellm is required for RULER evaluation")
    
    def _parse_ranking(self, response: str, n: int) -> List[int]:
        """Parse ranking from judge response.
        
        Expected format: RANKING: [3, 1, 4, 2]
        Returns 0-indexed ranking: [2, 0, 3, 1]
        """
        # Try to find RANKING: [...] pattern
        match = re.search(r'RANKING:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        
        if match:
            try:
                # Parse as comma-separated integers
                indices = [int(x.strip()) for x in match.group(1).split(',')]
                # Convert 1-indexed to 0-indexed
                ranking = [i - 1 for i in indices]
                
                # Validate
                if len(ranking) == n and set(ranking) == set(range(n)):
                    return ranking
            except ValueError:
                pass
        
        # Fallback: try to find any sequence of n numbers
        numbers = re.findall(r'\b(\d+)\b', response)
        for i in range(len(numbers) - n + 1):
            try:
                indices = [int(numbers[j]) for j in range(i, i + n)]
                ranking = [idx - 1 for idx in indices]
                if set(ranking) == set(range(n)):
                    return ranking
            except (ValueError, IndexError):
                continue
        
        # Final fallback: return natural order
        return list(range(n))
    
    def _format_example(self, example: Any) -> str:
        """Format an example for the prompt."""
        if isinstance(example, dict):
            return json.dumps(example, indent=2, default=str)
        return str(example)
    
    def _hash_candidate(self, candidate: Dict[str, str]) -> str:
        """Create a stable hash for a candidate configuration."""
        serialized = json.dumps(candidate, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
    
    def _cache_key(self, candidates: List[Dict[str, str]], example: Any) -> str:
        """Create cache key for a comparison."""
        parts = [self._hash_candidate(c) for c in candidates]
        parts.append(hashlib.md5(str(example).encode()).hexdigest()[:12])
        return ":".join(parts)
