"""Run a small end-to-end RULER-GEPA optimization loop on a toy support task.

Required environment:
    OPENAI_API_KEY

Optional environment:
    RULER_GEPA_GENERATION_MODEL
    RULER_GEPA_JUDGE_MODEL
    RULER_GEPA_MUTATION_MODEL
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from textwrap import indent
from typing import Any

import litellm

from ruler_gepa import RulerAdapter, RulerConfig, RulerGEPAEngine, build_reflection_payload

OBJECTIVE = (
    "Rank responses by empathy, clarity, actionability, and policy safety. "
    "Prefer concise replies under 80 words that explicitly acknowledge the issue "
    "and give the user a concrete next step."
)

TOY_DATASET = [
    {
        "customer_message": "I was charged twice for my subscription this month. Can you fix it?",
        "company_context": "SaaS billing support",
    },
    {
        "customer_message": "My order says delivered but I never got it.",
        "company_context": "E-commerce support",
    },
    {
        "customer_message": "The app crashes every time I upload a photo from my phone.",
        "company_context": "Consumer mobile app support",
    },
    {
        "customer_message": "I need to cancel before renewal, but I can't find the setting.",
        "company_context": "Subscription software support",
    },
    {
        "customer_message": "Your team still hasn't replied to my last two emails.",
        "company_context": "General customer support",
    },
]

SEED_PROMPT = """You are a customer support assistant.
Reply politely and try to help."""

BASELINE_CANDIDATES = [
    {
        "system_prompt": (
            "You are a customer support assistant. Be warm, concise, and practical. "
            "Acknowledge the problem, apologize when appropriate, and give a concrete next step."
        )
    },
    {
        "system_prompt": (
            "You are a support agent writing crisp replies. "
            "Stay under 80 words, avoid filler, and include the exact next action the customer should take."
        )
    },
]

MUTATION_TEMPLATE = """You are improving a system prompt for a toy customer support task.

## Objective
{objective}

## Current Prompt
```text
{current_prompt}
```

## Comparative Reflection
{reflection_prompt}

## Instructions
Produce one improved system prompt that should outperform the current one on the objective.
Keep it practical and specific.

Return exactly:
IMPROVED_PROMPT:
<prompt text>
"""


@dataclass
class EvalResult:
    outputs: list[str]


class ToySupportAdapter:
    """Minimal base adapter that renders outputs using an LLM."""

    def __init__(self, generation_model: str) -> None:
        self.generation_model = generation_model

    def evaluate(
        self,
        batch: list[dict[str, str]],
        candidate: dict[str, str],
        capture_traces: bool = True,
    ) -> EvalResult:
        outputs: list[str] = []
        for example in batch:
            messages = [
                {"role": "system", "content": candidate["system_prompt"]},
                {"role": "user", "content": self._build_user_prompt(example)},
            ]
            response = call_completion(
                model=self.generation_model,
                messages=messages,
                temperature=0.2,
            )
            outputs.append(response.choices[0].message.content.strip())
        return EvalResult(outputs=outputs)

    @staticmethod
    def _build_user_prompt(example: dict[str, str]) -> str:
        return (
            f"Context: {example['company_context']}\n"
            f"Customer message: {example['customer_message']}\n\n"
            "Write the best possible support response."
        )


def mutate_candidate(
    current_candidate: dict[str, str],
    comparison_results: list[dict[str, Any]],
    mutation_model: str,
) -> dict[str, str]:
    """Ask the mutation model to propose a better prompt."""
    reflection = build_reflection_payload(current_candidate, comparison_results)
    prompt = MUTATION_TEMPLATE.format(
        objective=OBJECTIVE,
        current_prompt=current_candidate["system_prompt"],
        reflection_prompt=reflection["prompt"],
    )
    response = call_completion(
        model=mutation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    mutated_text = response.choices[0].message.content.strip()
    return {"system_prompt": parse_improved_prompt(mutated_text)}


def parse_improved_prompt(text: str) -> str:
    """Extract the improved prompt from the mutation model response."""
    marker = "IMPROVED_PROMPT:"
    if marker in text:
        _, remainder = text.split(marker, 1)
        cleaned = remainder.strip()
        if cleaned:
            return cleaned
    return text.strip()


def collect_comparison_results(
    adapter: RulerAdapter,
    current_candidate: dict[str, str],
    comparison_pool: list[dict[str, str]],
    dataset: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Evaluate the current candidate against a comparison pool."""
    comparison_results: list[dict[str, Any]] = []
    for example in dataset:
        candidates = [current_candidate, *comparison_pool]
        ranking, metadata = adapter.relative_evaluate(candidates, example, rubric=OBJECTIVE)
        comparison_results.append(
            {
                "example": example,
                "ranking": ranking,
                "outputs": metadata["outputs"],
                "candidates": candidates,
            }
        )
    return comparison_results


def print_candidate(label: str, candidate: dict[str, str]) -> None:
    print(f"\n{label}")
    print(indent(candidate["system_prompt"], prefix="  "))


def run_optimization(iterations: int | None = None) -> None:
    generation_model = normalize_model_name(os.getenv("RULER_GEPA_GENERATION_MODEL", "gpt-5.3-mini"))
    judge_model = normalize_model_name(os.getenv("RULER_GEPA_JUDGE_MODEL", "gpt-5.3"))
    mutation_model = os.getenv("RULER_GEPA_MUTATION_MODEL", judge_model)
    mutation_model = normalize_model_name(mutation_model)
    iterations = iterations or int(os.getenv("RULER_GEPA_ITERATIONS", "3"))
    max_examples = int(os.getenv("RULER_GEPA_MAX_EXAMPLES", str(len(TOY_DATASET))))
    pool_size = int(os.getenv("RULER_GEPA_POOL_SIZE", str(len(BASELINE_CANDIDATES))))
    dataset = TOY_DATASET[:max_examples]

    adapter = RulerAdapter(
        base_adapter=ToySupportAdapter(generation_model=generation_model),
        config=RulerConfig(
            judge_lm=judge_model,
            rubric=OBJECTIVE,
            comparison_batch_size=3,
            cache_rankings=True,
        ),
    )
    engine = RulerGEPAEngine(adapter=adapter, trainset=dataset, rubric=OBJECTIVE)

    current_candidate = {"system_prompt": SEED_PROMPT}
    engine.register_candidate(current_candidate, accepted=True, metadata={"role": "seed"})

    print(f"Generation model: {generation_model}")
    print(f"Judge model: {judge_model}")
    print(f"Mutation model: {mutation_model}")
    print_candidate("Seed prompt:", current_candidate)

    comparison_pool = list(BASELINE_CANDIDATES[:pool_size])

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration} ===")
        comparison_results = collect_comparison_results(
            adapter=adapter,
            current_candidate=current_candidate,
            comparison_pool=comparison_pool,
            dataset=dataset,
        )
        proposed_candidate = mutate_candidate(current_candidate, comparison_results, mutation_model)
        print_candidate("Proposed prompt:", proposed_candidate)

        decision = engine.accept_candidate(
            new_candidate=proposed_candidate,
            parent_candidate=current_candidate,
            minibatch=dataset,
        )
        print(
            f"Accepted: {decision.accepted} | "
            f"wins={decision.wins}/{decision.total} | "
            f"win_rate={decision.win_rate:.2f}"
        )

        if decision.accepted:
            current_candidate = proposed_candidate

        dynamic_pool = [
            record.candidate
            for record in engine.select_frontier(limit=3)
            if record.candidate != current_candidate
        ]
        comparison_pool = (dynamic_pool + BASELINE_CANDIDATES)[:pool_size]

    print_candidate("Final prompt:", current_candidate)
    print("\nAdapter stats:")
    for key, value in adapter.stats.items():
        print(f"  {key}: {value}")


def ensure_secret_present() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise SystemExit(
        "Missing OPENAI_API_KEY. Set it in your shell or a local gitignored env file before running this example."
    )


def normalize_model_name(model: str) -> str:
    """Expand bare OpenAI model IDs to the provider-qualified format litellm expects."""
    if "/" in model:
        return model
    if os.getenv("OPENAI_API_KEY"):
        return f"openai/{model}"
    return model


def completion_kwargs_for_model(model: str, temperature: float) -> dict[str, float]:
    """Apply temperature only when the target model supports it."""
    if "gpt-5" in model.lower():
        return {}
    return {"temperature": temperature}


def call_completion(model: str, messages: list[dict[str, str]], temperature: float) -> Any:
    """Wrap litellm with basic rate-limit backoff."""
    kwargs = completion_kwargs_for_model(model, temperature)
    for attempt in range(5):
        try:
            return litellm.completion(model=model, messages=messages, **kwargs)
        except Exception as exc:
            delay = extract_retry_delay(str(exc))
            if delay is None or attempt == 4:
                raise
            print(f"Rate limited for {model}; sleeping {delay + 1:.0f}s before retry.")
            time.sleep(delay + 1)
    raise RuntimeError("Unreachable")


def extract_retry_delay(message: str) -> float | None:
    """Parse provider retry delays from error messages."""
    match = re.search(r"try again in ([0-9]+(?:\\.[0-9]+)?)s", message, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


if __name__ == "__main__":
    ensure_secret_present()
    try:
        run_optimization()
    except KeyboardInterrupt:
        sys.exit(130)
