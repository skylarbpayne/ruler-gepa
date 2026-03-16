# Contributing to RULER-GEPA

## Development Setup

```bash
# Clone the repo
git clone https://github.com/skylarbpayne/ruler-gepa.git
cd ruler-gepa

# Create/update the uv-managed environment
uv sync --dev

# Run tests
uv run pytest
```

## Project Status

This is currently a **research prototype**. We're validating the hypothesis that RULER-style relative evaluation can improve GEPA's prompt optimization.

## How to Contribute

### Discussing the Plan

The best way to contribute right now is reviewing and discussing the implementation plan:

1. Read [PLAN.md](./PLAN.md)
2. Open an issue with questions, concerns, or suggestions
3. Particularly valuable feedback on:
   - Open questions in Section 10
   - Ablation study design in Section 6
   - Benchmark selection in Section 5

### Code Contributions

The initial research scaffold is in place. Current focus areas:

1. **Engine integration:** Connect the prototype engine to upstream GEPA state/frontier management
2. **Enhanced reflection:** Expand mutation prompting around comparative datasets
3. **Benchmarks:** Wire up PAPILLON and IFBench
4. **Ablations:** Add experiment runners and cost tracking

Pick a phase that interests you and open an issue to coordinate.

## Code Style

- Use `ruff` for formatting and linting
- Type hints for all public functions
- Docstrings in Google style

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest tests/ --cov=src/ruler_gepa

# Type checking
uv run mypy src/ruler_gepa
```
