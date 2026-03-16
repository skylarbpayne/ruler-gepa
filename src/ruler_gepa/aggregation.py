"""Ranking aggregation methods for RULER-GEPA.

Converts pairwise/listwise rankings into candidate scores
suitable for Pareto frontier selection.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class BradleyTerryAggregator:
    """Bradley-Terry model for converting rankings to scores.
    
    The Bradley-Terry model assumes P(i beats j) = score_i / (score_i + score_j).
    We use an online update rule to incrementally update scores from observed rankings.
    
    Scores are maintained in log-odds space (like Elo ratings) for numerical stability.
    """
    
    learning_rate: float = 0.1
    initial_score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    comparison_count: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    def update_from_ranking(self, candidate_ids: List[str], ranking: List[int]) -> None:
        """Update scores from an observed ranking.
        
        Args:
            candidate_ids: List of candidate identifiers (hashes or indices).
            ranking: Indices into candidate_ids, ordered best to worst.
                     e.g., [2, 0, 1] means candidate_ids[2] is best.
        """
        # Initialize any new candidates
        for cid in candidate_ids:
            if cid not in self.scores:
                self.scores[cid] = self.initial_score
        
        # Extract pairwise comparisons from ranking
        for i, winner_idx in enumerate(ranking):
            winner_id = candidate_ids[winner_idx]
            
            for loser_idx in ranking[i + 1:]:
                loser_id = candidate_ids[loser_idx]
                self._update_pair(winner_id, loser_id)
    
    def _update_pair(self, winner: str, loser: str) -> None:
        """Update scores from a single pairwise comparison."""
        # Expected probability that winner beats loser (sigmoid of score diff)
        score_diff = self.scores[winner] - self.scores[loser]
        p_win = 1 / (1 + 10 ** (-score_diff / 400))  # Elo-style scaling
        
        # Update: winner won (actual = 1), so error = 1 - p_win
        update = self.learning_rate * (1 - p_win)
        self.scores[winner] += update
        self.scores[loser] -= update
        
        # Track comparison count
        pair = (min(winner, loser), max(winner, loser))
        self.comparison_count[pair] = self.comparison_count.get(pair, 0) + 1
    
    def get_score(self, candidate_id: str) -> float:
        """Get current score for a candidate."""
        return self.scores.get(candidate_id, self.initial_score)
    
    def get_win_probability(self, candidate_a: str, candidate_b: str) -> float:
        """Predicted probability that A beats B."""
        score_diff = self.get_score(candidate_a) - self.get_score(candidate_b)
        return 1 / (1 + 10 ** (-score_diff / 400))
    
    def get_ranking(self, candidate_ids: List[str]) -> List[str]:
        """Rank candidates by their Bradley-Terry scores (best first)."""
        return sorted(candidate_ids, key=lambda x: -self.get_score(x))


@dataclass
class EloAggregator:
    """Elo rating system for ranking aggregation.
    
    Similar to Bradley-Terry but with the standard Elo K-factor update.
    """
    
    k_factor: float = 32.0
    initial_rating: float = 1500.0
    ratings: Dict[str, float] = field(default_factory=dict)
    
    def update_from_ranking(self, candidate_ids: List[str], ranking: List[int]) -> None:
        """Update Elo ratings from observed ranking."""
        for cid in candidate_ids:
            if cid not in self.ratings:
                self.ratings[cid] = self.initial_rating
        
        for i, winner_idx in enumerate(ranking):
            winner_id = candidate_ids[winner_idx]
            for loser_idx in ranking[i + 1:]:
                loser_id = candidate_ids[loser_idx]
                self._update_pair(winner_id, loser_id)
    
    def _update_pair(self, winner: str, loser: str) -> None:
        """Standard Elo update for a single match."""
        expected_winner = 1 / (1 + 10 ** ((self.ratings[loser] - self.ratings[winner]) / 400))
        expected_loser = 1 - expected_winner
        
        self.ratings[winner] += self.k_factor * (1 - expected_winner)
        self.ratings[loser] += self.k_factor * (0 - expected_loser)
    
    def get_score(self, candidate_id: str) -> float:
        return self.ratings.get(candidate_id, self.initial_rating)


@dataclass 
class CopelandAggregator:
    """Copeland method: score = wins - losses.
    
    Simple and robust to intransitive preferences.
    """
    
    wins: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, int] = field(default_factory=dict)
    
    def update_from_ranking(self, candidate_ids: List[str], ranking: List[int]) -> None:
        """Update win/loss counts from ranking."""
        for cid in candidate_ids:
            self.wins.setdefault(cid, 0)
            self.losses.setdefault(cid, 0)
        
        for i, winner_idx in enumerate(ranking):
            winner_id = candidate_ids[winner_idx]
            # Winner beats everyone ranked below
            self.wins[winner_id] += len(ranking) - i - 1
            
            for loser_idx in ranking[i + 1:]:
                loser_id = candidate_ids[loser_idx]
                self.losses[loser_id] += 1
    
    def get_score(self, candidate_id: str) -> float:
        """Copeland score = wins - losses."""
        return self.wins.get(candidate_id, 0) - self.losses.get(candidate_id, 0)


def create_aggregator(method: str, **kwargs):
    """Factory function for aggregators."""
    if method == "bradley-terry":
        return BradleyTerryAggregator(**kwargs)
    elif method == "elo":
        return EloAggregator(**kwargs)
    elif method == "copeland":
        return CopelandAggregator()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
