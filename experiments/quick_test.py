#!/usr/bin/env python3
"""
Quick sanity check: GEPA vs RULER-GEPA on synthetic data.
Smaller scale (5 train, 3 val, 5 iterations) for fast turnaround.
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ruler_gepa.datasets import load_mini_ifbench

# Use cheaper/faster models
TASK_MODEL = os.environ.get("RULER_GEPA_GENERATION_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.environ.get("RULER_GEPA_JUDGE_MODEL", "gpt-4.1")
REFLECTION_MODEL = os.environ.get("RULER_GEPA_MUTATION_MODEL", "gpt-4.1")

# Smaller scale for quick test
MAX_ITERATIONS = 5
MINIBATCH_SIZE = 3
TRAIN_SIZE = 10
VAL_SIZE = 5

SEED_PROMPT = """You are a helpful assistant. Follow the user's instructions carefully and provide accurate, relevant responses."""


@dataclass
class Metrics:
    name: str
    scores: list[float] = field(default_factory=list)
    final_score: float = 0.0
    llm_calls: int = 0
    duration: float = 0.0
    final_prompt: str = ""


def lm_call(model: str, prompt: str) -> str:
    """Simple LLM call."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_response(system_prompt: str, user_input: str) -> str:
    """Generate response with system prompt."""
    response = litellm.completion(
        model=TASK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content


def judge_response(instruction: str, response: str, reference: str) -> tuple[float, str]:
    """Judge a response absolutely (0-1 score)."""
    prompt = f"""Rate this response to an instruction on a scale of 0-10.

INSTRUCTION:
{instruction}

REFERENCE ANSWER:
{reference}

ACTUAL RESPONSE:
{response}

Provide your rating as a single number (0-10) on the first line, then explain briefly.
"""
    result = lm_call(JUDGE_MODEL, prompt)
    try:
        score = float(result.strip().split()[0]) / 10.0
    except:
        score = 0.5
    return min(max(score, 0.0), 1.0), result


def compare_responses(instruction: str, response_a: str, response_b: str, reference: str) -> str:
    """RULER-style: which response is better? Returns 'A' or 'B'."""
    prompt = f"""Compare these two responses to the same instruction.

INSTRUCTION:
{instruction}

REFERENCE (for context):
{reference}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response better follows the instruction? Reply with ONLY "A" or "B".
"""
    result = lm_call(JUDGE_MODEL, prompt)
    return "B" if "B" in result.upper()[:5] else "A"


def mutate_prompt(current: str, feedback: list[dict]) -> str:
    """Generate improved prompt based on feedback."""
    feedback_str = "\n".join([
        f"Input: {f['input'][:100]}...\nScore: {f['score']:.2f}\nIssue: {f['feedback'][:200]}"
        for f in feedback[:3]
    ])
    
    prompt = f"""Improve this system prompt based on performance feedback.

CURRENT PROMPT:
{current}

PERFORMANCE FEEDBACK:
{feedback_str}

Write an improved prompt. Output ONLY the new prompt, no explanation:
"""
    result = lm_call(REFLECTION_MODEL, prompt)
    # Clean up
    result = result.strip()
    if result.startswith("```"):
        result = result.split("```")[1].strip()
    return result


def evaluate_batch(system_prompt: str, examples: list) -> list[dict]:
    """Evaluate prompt on examples, return scores + feedback."""
    results = []
    for ex in examples:
        response = generate_response(system_prompt, ex["input"])
        score, feedback = judge_response(ex["input"], response, ex["answer"])
        results.append({
            "input": ex["input"],
            "response": response,
            "score": score,
            "feedback": feedback,
        })
    return results


def run_baseline(examples: list) -> Metrics:
    """Evaluate seed prompt."""
    print("\n=== Baseline ===")
    start = time.time()
    results = evaluate_batch(SEED_PROMPT, examples)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Score: {avg:.3f}")
    return Metrics(
        name="baseline",
        scores=[avg],
        final_score=avg,
        duration=time.time() - start,
        final_prompt=SEED_PROMPT,
    )


def run_gepa(train: list, val: list) -> Metrics:
    """Standard GEPA with absolute scoring."""
    print("\n=== GEPA (Absolute) ===")
    start = time.time()
    current = SEED_PROMPT
    val_scores = []
    llm_calls = 0
    
    for i in range(MAX_ITERATIONS):
        print(f"  Iter {i+1}/{MAX_ITERATIONS}", end=" ")
        
        # Sample minibatch
        batch = random.sample(train, min(MINIBATCH_SIZE, len(train)))
        
        # Evaluate current
        results = evaluate_batch(current, batch)
        llm_calls += len(batch) * 2  # generate + judge
        train_score = sum(r["score"] for r in results) / len(results)
        
        # Mutate
        new_prompt = mutate_prompt(current, results)
        llm_calls += 1
        
        # Evaluate mutant
        new_results = evaluate_batch(new_prompt, batch)
        llm_calls += len(batch) * 2
        new_score = sum(r["score"] for r in new_results) / len(new_results)
        
        # Accept if better
        if new_score > train_score:
            current = new_prompt
            print(f"✓ {train_score:.2f}→{new_score:.2f}")
        else:
            print(f"✗ {train_score:.2f}>{new_score:.2f}")
        
        # Validate
        val_results = evaluate_batch(current, val)
        llm_calls += len(val) * 2
        val_score = sum(r["score"] for r in val_results) / len(val_results)
        val_scores.append(val_score)
    
    print(f"  Final val: {val_scores[-1]:.3f}")
    return Metrics(
        name="gepa",
        scores=val_scores,
        final_score=val_scores[-1],
        llm_calls=llm_calls,
        duration=time.time() - start,
        final_prompt=current,
    )


def run_ruler_gepa(train: list, val: list) -> Metrics:
    """RULER-GEPA with relative evaluation."""
    print("\n=== RULER-GEPA (Relative) ===")
    start = time.time()
    current = SEED_PROMPT
    val_scores = []
    llm_calls = 0
    
    for i in range(MAX_ITERATIONS):
        print(f"  Iter {i+1}/{MAX_ITERATIONS}", end=" ")
        
        # Sample minibatch
        batch = random.sample(train, min(MINIBATCH_SIZE, len(train)))
        
        # Evaluate current (for reflection feedback)
        results = evaluate_batch(current, batch)
        llm_calls += len(batch) * 2
        
        # Mutate
        new_prompt = mutate_prompt(current, results)
        llm_calls += 1
        
        # Generate mutant responses
        mutant_responses = [generate_response(new_prompt, ex["input"]) for ex in batch]
        llm_calls += len(batch)
        
        # RULER: Pairwise comparison
        wins = 0
        for j, ex in enumerate(batch):
            winner = compare_responses(
                ex["input"],
                results[j]["response"],  # A = current
                mutant_responses[j],      # B = mutant
                ex["answer"],
            )
            llm_calls += 1
            if winner == "B":
                wins += 1
        
        win_rate = wins / len(batch)
        
        # Accept if wins majority
        if win_rate > 0.5:
            current = new_prompt
            print(f"✓ winrate={win_rate:.0%}")
        else:
            print(f"✗ winrate={win_rate:.0%}")
        
        # Validate (for comparison only)
        val_results = evaluate_batch(current, val)
        llm_calls += len(val) * 2
        val_score = sum(r["score"] for r in val_results) / len(val_results)
        val_scores.append(val_score)
    
    print(f"  Final val: {val_scores[-1]:.3f}")
    return Metrics(
        name="ruler_gepa",
        scores=val_scores,
        final_score=val_scores[-1],
        llm_calls=llm_calls,
        duration=time.time() - start,
        final_prompt=current,
    )


def main():
    print("=" * 50)
    print("RULER-GEPA Quick Test")
    print("=" * 50)
    
    # Force synthetic data (no HF download)
    print("\nLoading synthetic data...")
    data = load_mini_ifbench(train_size=TRAIN_SIZE, val_size=VAL_SIZE, use_hf=False)
    print(f"  Train: {len(data.trainset)}, Val: {len(data.valset)}, Source: {data.source}")
    
    print(f"\nModels: task={TASK_MODEL}, judge={JUDGE_MODEL}, reflect={REFLECTION_MODEL}")
    print(f"Config: {MAX_ITERATIONS} iters, minibatch={MINIBATCH_SIZE}")
    
    # Run experiments
    baseline = run_baseline(data.valset)
    gepa = run_gepa(data.trainset, data.valset)
    ruler = run_ruler_gepa(data.trainset, data.valset)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Baseline:    {baseline.final_score:.3f}")
    print(f"  GEPA:        {gepa.final_score:.3f} ({gepa.final_score - baseline.final_score:+.3f})")
    print(f"  RULER-GEPA:  {ruler.final_score:.3f} ({ruler.final_score - baseline.final_score:+.3f})")
    print(f"\n  GEPA calls:       {gepa.llm_calls}")
    print(f"  RULER-GEPA calls: {ruler.llm_calls}")
    
    # Success check
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA")
    print("=" * 50)
    
    ruler_vs_gepa = ruler.final_score - gepa.final_score
    within_5pct = abs(ruler_vs_gepa) < 0.05
    ruler_improves = ruler.final_score > baseline.final_score
    
    print(f"  {'✓' if ruler_improves else '✗'} RULER-GEPA improves over baseline")
    print(f"  {'✓' if within_5pct else '✗'} RULER-GEPA within 5% of GEPA (diff={ruler_vs_gepa:+.3f})")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": asdict(baseline),
        "gepa": asdict(gepa),
        "ruler_gepa": asdict(ruler),
        "success": ruler_improves and within_5pct,
    }
    
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"quick_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out_file}")
    
    return 0 if (ruler_improves and within_5pct) else 1


if __name__ == "__main__":
    sys.exit(main())
