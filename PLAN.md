# RULER-GEPA Implementation Plan

## 1. Background

### GEPA (Genetic-Pareto)

GEPA is a prompt optimization framework that:
- Evolves prompts through LLM-based reflection and mutation
- Uses Pareto-efficient selection (candidates excelling on different subsets survive)
- Achieves SOTA results with 35x fewer evaluations than RL methods

**Core loop:**
```
1. Select candidate from Pareto frontier
2. Evaluate on minibatch → get scores + execution traces
3. Reflect on traces → diagnose failures
4. Mutate → propose improved candidate
5. Accept if improved on subsample
6. Update Pareto frontier
```

**Key concept: Actionable Side Information (ASI)**
- Diagnostic feedback returned by evaluators
- Error messages, profiler output, reasoning logs
- The "gradient" for text optimization

### RULER (Relative Universal LLM-Elicited Rewards)

RULER simplifies RL reward design by:
- Using LLM-as-judge to rank trajectories (not score them absolutely)
- Presenting multiple candidates simultaneously for comparison
- Focusing on relative ranking to avoid calibration issues

**Core mechanism:**
```
1. Generate N trajectories for same input
2. Deduplicate common prefixes
3. Present all N to LLM-as-judge with rubric
4. Judge ranks them 0-1 based on success
5. Use rankings as reward signal
```

---

## 2. The Synthesis: RULER-GEPA

### Key Idea

Replace GEPA's absolute `evaluate(candidate) → float` with relative evaluation:

```python
# GEPA today
evaluate(candidate_A, example) → 0.73
evaluate(candidate_B, example) → 0.68

# RULER-GEPA
relative_evaluate([A, B, C, D], example) → ranking [A, C, B, D]
```

### What Changes

| Component | GEPA Today | RULER-GEPA |
|-----------|------------|------------|
| **Evaluation** | Absolute scores per candidate | LLM-as-judge ranks candidates |
| **Acceptance** | `new_score > old_score` | `new beats old on >50% examples` |
| **Pareto frontier** | Track scores per task | Track Bradley-Terry ratings |
| **Reflection input** | "Scored 0.73 on task 1" | "Beat B on task 1, lost to C on task 2" |
| **Reflection output** | Generic improvement | Targeted: "close gap to C on formatting" |

### Expected Benefits

1. **No calibration needed** — "Is A better than B?" is more reliable
2. **Richer signal** — Comparisons contain dimensional information
3. **Works when absolute scoring is hard** — Subjective tasks
4. **Potentially more efficient** — One call ranks 4+ candidates

### Challenges

1. **Transitivity** — A>B, B>C doesn't guarantee A>C with LLM judges
2. **Combinatorics** — n candidates × m examples = lots of comparisons
3. **Pareto adaptation** — Need to convert rankings to selection scores

---

## 3. Architecture

### Integration Point

GEPA has a clean adapter pattern. We create a `RulerAdapter` that wraps any base adapter:

```
optimize_anything API
        ↓
    GEPAEngine
        ↓
    RulerAdapter (NEW)
        ↓
    BaseAdapter (existing)
        ↓
    GEPAState (modified for BT scores)
```

### Core Components

```
src/ruler_gepa/
├── adapter.py          # RulerAdapter wrapping base adapters
├── engine.py           # Modified GEPAEngine with relative eval
├── proposer.py         # Enhanced reflection with comparisons
├── aggregation.py      # Bradley-Terry / Elo score computation
├── prompts.py          # Ranking prompt templates
└── config.py           # RulerConfig dataclass
```

---

## 4. Implementation Details

### 4.1 RulerAdapter

```python
class RulerAdapter(GEPAAdapter):
    """Wraps any adapter with RULER-style relative evaluation."""
    
    def __init__(
        self,
        base_adapter: GEPAAdapter,
        judge_lm: str = "openai/gpt-4.1",
        comparison_batch_size: int = 4,
        rubric: str | None = None,
    ):
        self.base = base_adapter
        self.judge_lm = judge_lm
        self.comparison_batch_size = comparison_batch_size
        self.rubric = rubric
        self.bt_scores: Dict[str, float] = {}
    
    def evaluate(self, batch, candidate, capture_traces=True):
        """Standard eval for trace capture (reflection needs this)."""
        return self.base.evaluate(batch, candidate, capture_traces)
    
    def relative_evaluate(
        self,
        candidates: List[Dict[str, str]],
        example: Any,
    ) -> List[int]:
        """
        Rank candidates on single example using LLM-as-judge.
        Returns indices from best to worst.
        """
        # 1. Run each candidate
        outputs = [self.base.evaluate([example], c, False).outputs[0] 
                   for c in candidates]
        
        # 2. Build ranking prompt
        prompt = self._build_ranking_prompt(example, candidates, outputs)
        
        # 3. Call judge
        response = litellm.completion(model=self.judge_lm, ...)
        
        # 4. Parse ranking
        return self._parse_ranking(response)
```

### 4.2 Ranking Prompt (RULER-style)

```python
RANKING_PROMPT = """You are evaluating {n} approaches to a task.

## Task Input
{input}

## Evaluation Criteria
{rubric}

## Candidates

{candidates_section}

## Instructions
Rank all candidates from BEST (1) to WORST ({n}).

Think step by step:
1. For each candidate, assess strengths and weaknesses
2. Compare pairs where ranking is unclear
3. Produce final ranking

Output format: RANKING: [best_idx, second_best_idx, ..., worst_idx]

Your analysis and ranking:"""
```

### 4.3 Bradley-Terry Score Updates

```python
class BradleyTerryAggregator:
    """Convert pairwise rankings to scores."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.scores: Dict[str, float] = {}
        self.lr = learning_rate
    
    def update(self, candidate_ids: List[str], ranking: List[int]):
        """Update scores from observed ranking."""
        for i, winner_idx in enumerate(ranking):
            winner = candidate_ids[winner_idx]
            self.scores.setdefault(winner, 0.0)
            
            for loser_idx in ranking[i+1:]:
                loser = candidate_ids[loser_idx]
                self.scores.setdefault(loser, 0.0)
                
                # Expected win probability
                p_win = 1 / (1 + 10**((self.scores[loser] - self.scores[winner]) / 400))
                
                # Update (winner won, actual=1)
                self.scores[winner] += self.lr * (1 - p_win)
                self.scores[loser] -= self.lr * (1 - p_win)
    
    def get_score(self, candidate_id: str) -> float:
        return self.scores.get(candidate_id, 0.0)
```

### 4.4 Modified Acceptance Criteria

```python
def _accept_candidate(
    self,
    new_candidate: Dict[str, str],
    parent_candidate: Dict[str, str],
    minibatch: List[Any],
) -> Tuple[bool, Dict]:
    """
    RULER-style acceptance: compare against parent on minibatch.
    Accept if new candidate wins on majority of examples.
    """
    wins = 0
    comparisons = []
    
    for example in minibatch:
        ranking = self.ruler_adapter.relative_evaluate(
            [parent_candidate, new_candidate],
            example
        )
        
        new_wins = ranking[0] == 1  # new_candidate is index 1
        if new_wins:
            wins += 1
        
        comparisons.append({
            "example": example,
            "new_wins": new_wins,
            "ranking": ranking,
        })
    
    win_rate = wins / len(minibatch)
    accepted = win_rate > 0.5
    
    return accepted, {
        "win_rate": win_rate,
        "wins": wins,
        "total": len(minibatch),
        "comparisons": comparisons,
    }
```

### 4.5 Enhanced Reflection with Comparative Feedback

```python
def build_comparative_reflective_dataset(
    candidate: Dict[str, str],
    comparison_results: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Build reflective dataset with comparative feedback.
    
    Instead of: "Scored 0.73"
    We provide: "Beat B on reasoning, lost to C on formatting"
    """
    dataset = {}
    
    for result in comparison_results:
        entry = {
            "input": result["example"],
            "our_output": result["outputs"][0],
            "our_rank": result["ranking"].index(0) + 1,
            
            # What we beat
            "beat": [
                {"approach": summarize(c), "output": o}
                for i, (c, o) in enumerate(zip(result["candidates"], result["outputs"]))
                if result["ranking"].index(i) > result["ranking"].index(0)
            ],
            
            # What beat us
            "lost_to": [
                {"approach": summarize(c), "output": o}
                for i, (c, o) in enumerate(zip(result["candidates"], result["outputs"]))
                if result["ranking"].index(i) < result["ranking"].index(0)
            ],
        }
        dataset.setdefault("main", []).append(entry)
    
    return dataset
```

### 4.6 Comparative Reflection Prompt

```python
COMPARATIVE_REFLECTION_PROMPT = """You are improving a prompt based on comparative evaluation.

## Current Prompt
{current_prompt}

## Comparative Results

{for each comparison}
### Example: {input}
**Our rank:** {rank} of {total}

**Approaches that beat us:**
{for winner in lost_to}
- Approach: {winner.approach}
  Output: {winner.output[:300]}...
{end for}

**Approaches we beat:**
{for loser in beat}
- Approach: {loser.approach}
  Output: {loser.output[:300]}...
{end for}
{end for}

## Analysis
1. What specific qualities do winning approaches have that we lack?
2. What pattern explains when we win vs lose?
3. What single change would most improve our ranking?

## Improved Prompt
Based on your analysis, write an improved prompt:"""
```

---

## 5. Test Benchmarks

From GEPA paper experiments:

| Benchmark | Type | Train/Val | Good for RULER? | Why |
|-----------|------|-----------|-----------------|-----|
| **PAPILLON** | Structured extraction | 326/326 | ✅ Excellent | Already uses LLM-as-judge |
| **IFBench** | Instruction following | 200/200 | ✅ Excellent | Multi-aspect evaluation |
| **HotPotQA** | Multi-hop QA | 500/500 | ✅ Good | Clear answer comparison |
| **Hover** | Fact verification | 500/500 | ⚠️ Maybe | Binary outcome limits comparison |
| **AIME** | Competition math | 90/30 | ⚠️ Limited | Binary correctness |
| **LiveBench Math** | Math reasoning | 100/100 | ⚠️ Limited | Binary correctness |

**Recommended order:**
1. PAPILLON — Easiest integration (already LLM-judged)
2. IFBench — Rich multi-aspect signal
3. HotPotQA — Test on reasoning tasks

---

## 6. Ablation Study Design

### Main Comparison

| Method | Description |
|--------|-------------|
| **GEPA** | Baseline with absolute scoring |
| **RULER-GEPA** | Full RULER integration |

### Ablations

| Ablation | What's Changed | What It Tests |
|----------|---------------|---------------|
| **Abl-RelativeOnly** | Relative eval, standard reflection | Value of relative scoring alone |
| **Abl-ReflectionOnly** | Absolute eval, comparative reflection | Value of comparative reflection alone |
| **Abl-NoBradleyTerry** | Win counting instead of BT | Value of proper ranking aggregation |
| **Abl-PairwiseOnly** | Only compare vs parent | Value of multi-candidate comparison |
| **Abl-LargerBatch** | Compare 8 candidates vs 4 | Batch size sensitivity |

### Metrics

1. **Final valset accuracy** — Primary metric
2. **Sample efficiency** — Evals to reach X% accuracy
3. **Reflection quality** — Human eval of mutation suggestions
4. **Judge consistency** — Transitivity violation rate
5. **Cost** — Total API spend (judge calls are expensive)

---

## 7. Configuration

```python
@dataclass
class RulerConfig:
    """Configuration for RULER-style evaluation."""
    
    # Core settings
    enabled: bool = True
    judge_lm: str = "openai/gpt-4.1"
    comparison_batch_size: int = 4
    
    # Rubric
    rubric: str | None = None
    use_objective_as_rubric: bool = True
    
    # Aggregation
    aggregation: Literal["bradley-terry", "elo", "copeland"] = "bradley-terry"
    bt_learning_rate: float = 0.1
    
    # Acceptance
    min_win_rate: float = 0.5
    
    # Efficiency
    deduplicate_common_prefix: bool = True
    cache_rankings: bool = True
```

---

## 8. Usage

### With optimize_anything

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig
from ruler_gepa import RulerConfig

result = optimize_anything(
    seed_candidate={"system_prompt": "You are a helpful assistant..."},
    evaluator=my_evaluator,
    dataset=trainset,
    valset=valset,
    config=GEPAConfig(
        evaluation_strategy=RulerConfig(
            judge_lm="openai/gpt-4.1",
            comparison_batch_size=4,
            rubric="Rank by helpfulness, accuracy, and clarity.",
        ),
    ),
)
```

### Standalone

```python
from ruler_gepa import RulerGEPAEngine, RulerAdapter

adapter = RulerAdapter(
    base_adapter=YourAdapter(),
    judge_lm="openai/gpt-4.1",
)

engine = RulerGEPAEngine(
    adapter=adapter,
    trainset=trainset,
    valset=valset,
    ...
)

result = engine.run()
```

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] `RulerAdapter` with `relative_evaluate()`
- [ ] Ranking prompt templates
- [ ] Bradley-Terry aggregation
- [ ] Basic tests

### Phase 2: Engine Integration  
- [ ] Modified acceptance criteria
- [ ] Pareto frontier with BT scores
- [ ] State serialization updates

### Phase 3: Enhanced Reflection
- [ ] Comparative reflective dataset builder
- [ ] Comparative reflection prompts
- [ ] Integration with mutation proposer

### Phase 4: Experiments
- [ ] PAPILLON benchmark integration
- [ ] IFBench benchmark integration
- [ ] Ablation study runs
- [ ] Results analysis

### Phase 5: Optimization
- [ ] Common prefix deduplication
- [ ] Ranking cache
- [ ] Cost tracking

---

## 10. Open Questions

1. **Judge model selection** — Should judge be same as reflection LM?
2. **Comparison pool** — Include random candidates or just Pareto frontier?
3. **Transitivity handling** — How to handle A>B>C but C>A?
4. **Cost tradeoff** — Is ranking 4 candidates cheaper than 4 absolute evals?
5. **Multi-objective** — How to handle multiple rubric dimensions?

---

## 11. References

- [GEPA Paper](https://arxiv.org/abs/2507.19457) — Agrawal et al., ICLR 2026 Oral
- [RULER Blog](https://openpipe.ai/blog/ruler) — OpenPipe
- [GEPA GitHub](https://github.com/gepa-ai/gepa)
- [optimize_anything Blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)
- [Bradley-Terry Model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
