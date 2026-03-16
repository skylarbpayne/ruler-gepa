"""Tests for prompt and reflection helpers."""

from ruler_gepa.prompts import build_comparative_reflection_prompt, build_ranking_prompt
from ruler_gepa.proposer import build_comparative_reflective_dataset


def test_build_ranking_prompt_shows_only_differences_when_deduplicating():
    prompt = build_ranking_prompt(
        input_text="Solve the task",
        candidates=[
            {"system": "common", "style": "formal"},
            {"system": "common", "style": "direct"},
        ],
        outputs=["formal output", "direct output"],
        rubric="Rank by clarity.",
        deduplicate=True,
    )

    assert "style: formal" in prompt
    assert "style: direct" in prompt
    assert "system: common" not in prompt


def test_build_comparative_dataset_and_prompt():
    dataset = build_comparative_reflective_dataset(
        candidate={"prompt": "ours"},
        comparison_results=[
            {
                "example": {"input": "x"},
                "ranking": [1, 0, 2],
                "outputs": ["ours output", "winner output", "loser output"],
                "candidates": [
                    {"prompt": "ours"},
                    {"prompt": "winner"},
                    {"prompt": "loser"},
                ],
            }
        ],
    )

    assert dataset["main"][0]["our_rank"] == 2
    assert dataset["main"][0]["lost_to"][0]["approach"].startswith("prompt=winner")
    assert dataset["main"][0]["beat"][0]["approach"].startswith("prompt=loser")

    prompt = build_comparative_reflection_prompt(
        current_artifact="ours",
        comparison_results=[
            {
                "example": {"input": "x"},
                "ranking": [1, 0, 2],
                "outputs": ["ours output", "winner output", "loser output"],
                "candidates": [
                    {"prompt": "ours"},
                    {"prompt": "winner"},
                    {"prompt": "loser"},
                ],
            }
        ],
    )

    assert "Approaches that beat us" in prompt
    assert "winner output" in prompt
