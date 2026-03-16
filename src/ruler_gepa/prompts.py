"""Prompt templates for RULER-GEPA ranking and reflection."""

RANKING_PROMPT = """You are evaluating {n} different approaches to a task.

## Task Input
{input}

## Evaluation Criteria
{rubric}

## Candidates and Their Outputs

{candidates_section}

## Your Task

Rank all {n} candidates from BEST (rank 1) to WORST (rank {n}).

Think step by step:
1. For each candidate, identify key strengths and weaknesses
2. Compare candidates on the evaluation criteria
3. Resolve any ties by considering secondary factors

Output your final ranking in this exact format:
RANKING: [best_idx, second_idx, ..., worst_idx]

Where each number is the candidate index (1-{n}).

Example for 4 candidates where Candidate 3 is best:
RANKING: [3, 1, 4, 2]

Your analysis and ranking:"""


CANDIDATE_SECTION_TEMPLATE = """### Candidate {idx}
**Approach:**
```
{approach}
```

**Output:**
```
{output}
```
"""


COMPARATIVE_REFLECTION_PROMPT = """You are improving a text artifact based on comparative evaluation against other approaches.

## Current Artifact
```
{current_artifact}
```

## Comparative Evaluation Results

{comparisons_section}

## Analysis Questions

Based on the comparisons above:

1. **Winning patterns:** What specific qualities do approaches that beat us have that we lack?

2. **Losing patterns:** What do we do well that weaker approaches fail at?

3. **Key dimension:** What single dimension (clarity, specificity, structure, etc.) most explains our wins/losses?

4. **Targeted improvement:** What specific change would most improve our ranking against the approaches that beat us?

## Improved Artifact

Write an improved version that addresses the gaps you identified. Be specific and targeted—don't just make it "better overall."

Improved artifact:"""


COMPARISON_ENTRY_TEMPLATE = """### Example: {input_summary}

**Our rank:** {our_rank} of {total_candidates}
**Our output:** {our_output_preview}

**Approaches that beat us:**
{winners_section}

**Approaches we beat:**
{losers_section}
"""


WINNER_ENTRY = """- **{approach_summary}**
  Output: {output_preview}
  Why better: [Judge's implicit reasoning from ranking]
"""


LOSER_ENTRY = """- **{approach_summary}**
  Output: {output_preview}
"""


def build_ranking_prompt(
    input_text: str,
    candidates: list[dict[str, str]],
    outputs: list[str],
    rubric: str,
    deduplicate: bool = True,
) -> str:
    """Build a complete ranking prompt.
    
    Args:
        input_text: The task input being evaluated.
        candidates: List of candidate configurations (dicts of component -> text).
        outputs: List of outputs from running each candidate.
        rubric: Evaluation criteria.
        deduplicate: If True, show only differing parts of candidates.
    
    Returns:
        Complete prompt string for LLM judge.
    """
    n = len(candidates)
    
    # Build candidates section
    sections = []
    for i, (candidate, output) in enumerate(zip(candidates, outputs)):
        approach_text = _format_candidate(candidate, deduplicate, candidates)
        section = CANDIDATE_SECTION_TEMPLATE.format(
            idx=i + 1,
            approach=approach_text,
            output=_truncate(output, 500),
        )
        sections.append(section)
    
    candidates_section = "\n".join(sections)
    
    return RANKING_PROMPT.format(
        n=n,
        input=_truncate(input_text, 1000),
        rubric=rubric,
        candidates_section=candidates_section,
    )


def build_comparative_reflection_prompt(
    current_artifact: str,
    comparison_results: list[dict],
    max_comparisons: int = 5,
) -> str:
    """Build reflection prompt with comparative feedback.
    
    Args:
        current_artifact: The artifact being improved.
        comparison_results: List of comparison dicts with ranking info.
        max_comparisons: Maximum examples to include.
    
    Returns:
        Complete reflection prompt.
    """
    entries = []
    
    for result in comparison_results[:max_comparisons]:
        our_rank = result["ranking"].index(0) + 1
        total = len(result["ranking"])
        
        # Build winners section
        winners = []
        for idx in result["ranking"][:result["ranking"].index(0)]:
            if idx != 0:
                winners.append(WINNER_ENTRY.format(
                    approach_summary=_summarize_approach(result["candidates"][idx]),
                    output_preview=_truncate(result["outputs"][idx], 200),
                ))
        winners_section = "\n".join(winners) if winners else "(None - we ranked first)"
        
        # Build losers section
        losers = []
        for idx in result["ranking"][result["ranking"].index(0) + 1:]:
            losers.append(LOSER_ENTRY.format(
                approach_summary=_summarize_approach(result["candidates"][idx]),
                output_preview=_truncate(result["outputs"][idx], 200),
            ))
        losers_section = "\n".join(losers) if losers else "(None - we ranked last)"
        
        entry = COMPARISON_ENTRY_TEMPLATE.format(
            input_summary=_truncate(str(result["example"]), 100),
            our_rank=our_rank,
            total_candidates=total,
            our_output_preview=_truncate(result["outputs"][0], 200),
            winners_section=winners_section,
            losers_section=losers_section,
        )
        entries.append(entry)
    
    comparisons_section = "\n---\n".join(entries)
    
    return COMPARATIVE_REFLECTION_PROMPT.format(
        current_artifact=current_artifact,
        comparisons_section=comparisons_section,
    )


def _format_candidate(
    candidate: dict[str, str],
    deduplicate: bool,
    all_candidates: list[dict[str, str]],
) -> str:
    """Format a candidate, optionally showing only unique parts."""
    if not deduplicate or len(all_candidates) <= 1:
        return "\n".join(f"{k}: {v}" for k, v in candidate.items())
    
    # Find common values across all candidates
    common_keys = set(candidate.keys())
    for other in all_candidates:
        common_keys &= set(other.keys())
    
    # Show only differing parts
    unique_parts = {}
    for k in candidate:
        values = [c.get(k) for c in all_candidates]
        if len(set(values)) > 1:  # Not all same
            unique_parts[k] = candidate[k]
    
    if not unique_parts:
        return "(identical to other candidates)"
    
    return "\n".join(f"{k}: {v}" for k, v in unique_parts.items())


def _summarize_approach(candidate: dict[str, str]) -> str:
    """Create a brief summary of a candidate approach."""
    # Take first 100 chars of concatenated values
    full = " | ".join(f"{k}={v[:50]}" for k, v in candidate.items())
    return _truncate(full, 150)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
