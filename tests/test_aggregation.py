"""Tests for ranking aggregation methods."""

import pytest
from ruler_gepa.aggregation import BradleyTerryAggregator, EloAggregator, CopelandAggregator


class TestBradleyTerryAggregator:
    """Tests for Bradley-Terry score aggregation."""
    
    def test_initial_scores_equal(self):
        """New candidates should have equal initial scores."""
        bt = BradleyTerryAggregator()
        
        bt.update_from_ranking(["a", "b", "c"], [0, 1, 2])
        
        # After one ranking, winner should be higher
        assert bt.get_score("a") > bt.get_score("b")
        assert bt.get_score("b") > bt.get_score("c")
    
    def test_consistent_winner_dominates(self):
        """A candidate that always wins should have highest score."""
        bt = BradleyTerryAggregator()
        
        # "a" always beats "b" and "c"
        for _ in range(10):
            bt.update_from_ranking(["a", "b", "c"], [0, 1, 2])
        
        assert bt.get_score("a") > bt.get_score("b")
        assert bt.get_score("a") > bt.get_score("c")
    
    def test_win_probability_prediction(self):
        """Win probability should reflect observed results."""
        bt = BradleyTerryAggregator()
        
        # "a" beats "b" every time
        for _ in range(20):
            bt.update_from_ranking(["a", "b"], [0, 1])
        
        # Should predict high probability of a winning
        prob = bt.get_win_probability("a", "b")
        assert prob > 0.9
    
    def test_get_ranking(self):
        """get_ranking should return candidates sorted by score."""
        bt = BradleyTerryAggregator()
        
        bt.update_from_ranking(["a", "b", "c"], [2, 0, 1])  # c > a > b
        
        ranking = bt.get_ranking(["a", "b", "c"])
        assert ranking[0] == "c"  # Best
        assert ranking[-1] == "b"  # Worst


class TestEloAggregator:
    """Tests for Elo rating aggregation."""
    
    def test_winner_gains_rating(self):
        """Winner should gain rating, loser should lose."""
        elo = EloAggregator(initial_rating=1500)
        
        initial_a = elo.get_score("a")
        initial_b = elo.get_score("b")
        
        elo.update_from_ranking(["a", "b"], [0, 1])  # a beats b
        
        assert elo.get_score("a") > initial_a
        assert elo.get_score("b") < initial_b
    
    def test_rating_sum_preserved(self):
        """Total rating should be approximately preserved."""
        elo = EloAggregator(initial_rating=1500)
        
        candidates = ["a", "b", "c"]
        initial_sum = sum(elo.get_score(c) for c in candidates)
        
        elo.update_from_ranking(candidates, [0, 1, 2])
        
        final_sum = sum(elo.get_score(c) for c in candidates)
        # Sum changes slightly due to pairwise updates, but should be close
        assert abs(final_sum - initial_sum) < 100


class TestCopelandAggregator:
    """Tests for Copeland method aggregation."""
    
    def test_win_counting(self):
        """Should count wins correctly."""
        copeland = CopelandAggregator()
        
        # a beats b beats c
        copeland.update_from_ranking(["a", "b", "c"], [0, 1, 2])
        
        # a has 2 wins (beat b and c)
        # b has 1 win (beat c), 1 loss (to a)
        # c has 0 wins, 2 losses
        assert copeland.get_score("a") == 2  # 2 wins - 0 losses
        assert copeland.get_score("b") == 0  # 1 win - 1 loss
        assert copeland.get_score("c") == -2  # 0 wins - 2 losses
    
    def test_handles_ties(self):
        """Should handle multiple rankings."""
        copeland = CopelandAggregator()
        
        # Mixed results
        copeland.update_from_ranking(["a", "b"], [0, 1])  # a > b
        copeland.update_from_ranking(["a", "b"], [1, 0])  # b > a
        
        # Should be tied
        assert copeland.get_score("a") == copeland.get_score("b")
