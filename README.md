# RULER-GEPA

**Relative evaluation for prompt optimization**

RULER-GEPA combines two techniques:
- **[GEPA](https://github.com/gepa-ai/gepa)** — Genetic-Pareto prompt optimization using LLM reflection
- **[RULER](https://openpipe.ai/blog/ruler)** — Relative Universal LLM-Elicited Rewards

## The Hypothesis

GEPA requires defining metrics that return absolute scores (0.0 to 1.0). For many tasks, this is hard:
- "Rate this response quality from 0-10" — calibration varies wildly
- "Is this code correct?" — binary loses nuance
- "How good is this writing?" — entirely subjective

RULER's core insight: **"Is A better than B?" is easier to answer than "How good is A?"**

What if we use RULER-style relative evaluation inside GEPA's evolutionary loop?

## Expected Benefits

1. **No metric calibration needed** — Ranking sidesteps "what does 0.7 mean?"
2. **Richer reflection signal** — "You lost to X because of Y" is more actionable than "You scored 0.73"
3. **Works for subjective tasks** — Writing quality, UX copy, creative generation
4. **Batch efficiency** — One LLM call can rank 4-8 candidates simultaneously

## Project Status

🚧 **Research prototype** — Not production ready

Implemented in this initial repo:
- `RulerAdapter` for relative evaluation with optional judge injection
- Ranking aggregation via Bradley-Terry, Elo, and Copeland methods
- `RulerGEPAEngine` with majority-win acceptance logic, frontier tracking, and state export/import
- Comparative reflection dataset and prompt helpers for mutation workflows
- Ranking-cache persistence and lightweight cost/stats tracking
- Benchmark registry and ablation-plan scaffolding for PAPILLON, IFBench, and HotPotQA
- Basic prompt, adapter, engine, and aggregation tests

See [PLAN.md](./PLAN.md) for the complete implementation plan.

## Quick Start

```bash
uv sync --dev
uv run pytest
```

```python
from ruler_gepa import RulerAdapter, RulerConfig, RulerGEPAEngine

adapter = RulerAdapter(
    base_adapter=my_base_adapter,
    config=RulerConfig(
        judge_lm="openai/gpt-4.1",
        comparison_batch_size=4,
        rubric="Rank by correctness, clarity, and usefulness.",
    ),
)

engine = RulerGEPAEngine(adapter=adapter, rubric="Prefer the more useful answer.")
decision = engine.accept_candidate(new_candidate, parent_candidate, minibatch)
```

## Quick Links

- [Implementation Plan](./PLAN.md)
- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [RULER Blog Post](https://openpipe.ai/blog/ruler)

## License

MIT
