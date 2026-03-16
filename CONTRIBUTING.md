# Contributing to RULER-GEPA

## Development Setup

```bash
# Clone the repo
git clone https://github.com/skylarbpayne/ruler-gepa.git
cd ruler-gepa

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
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

Once the plan is finalized, we'll be implementing in phases:

1. **Phase 1:** Core adapter integration
2. **Phase 2:** Enhanced reflection
3. **Phase 3:** Benchmark experiments
4. **Phase 4:** Ablations

Pick a phase that interests you and open an issue to coordinate.

## Code Style

- Use `ruff` for formatting and linting
- Type hints for all public functions
- Docstrings in Google style

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/ruler_gepa

# Type checking
mypy src/ruler_gepa
```
