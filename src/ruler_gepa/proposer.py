"""Comparative reflection helpers for RULER-GEPA."""

from typing import Any

from ruler_gepa.prompts import build_comparative_reflection_prompt


def summarize_candidate(candidate: dict[str, str], max_len: int = 160) -> str:
    """Create a short human-readable summary for a candidate."""
    parts = [f"{key}={value}" for key, value in candidate.items()]
    text = " | ".join(parts)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_comparative_reflective_dataset(
    candidate: dict[str, str],
    comparison_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build a dataset capturing who beat us and who we beat."""
    entries: list[dict[str, Any]] = []
    candidate_summary = summarize_candidate(candidate)

    for result in comparison_results:
        ranking: list[int] = result["ranking"]
        our_position = ranking.index(0)
        outputs: list[str] = result["outputs"]
        candidates: list[dict[str, str]] = result["candidates"]

        beat = [
            {
                "approach": summarize_candidate(candidates[idx]),
                "output": outputs[idx],
            }
            for idx in ranking[our_position + 1 :]
        ]
        lost_to = [
            {
                "approach": summarize_candidate(candidates[idx]),
                "output": outputs[idx],
            }
            for idx in ranking[:our_position]
        ]

        entries.append(
            {
                "input": result["example"],
                "candidate_summary": candidate_summary,
                "our_output": outputs[0],
                "our_rank": our_position + 1,
                "total_candidates": len(ranking),
                "beat": beat,
                "lost_to": lost_to,
            }
        )

    return {"main": entries}


def build_reflection_payload(
    candidate: dict[str, str],
    comparison_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build both the structured dataset and the reflection prompt."""
    artifact = "\n".join(f"{key}: {value}" for key, value in candidate.items())
    dataset = build_comparative_reflective_dataset(candidate, comparison_results)
    prompt = build_comparative_reflection_prompt(
        current_artifact=artifact,
        comparison_results=comparison_results,
    )
    return {
        "dataset": dataset,
        "prompt": prompt,
    }
