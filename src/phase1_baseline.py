"""
phase1_baseline.py — Run all questions WITHOUT hints.

For each model, runs every question N_SAMPLES times at temperature 1.0.
Identifies which questions the model gets wrong (these are the ones
where hints can influence behavior).

Also saves the full baseline answer distribution, which we need for
computing hint lift and entropy change later.

Features:
- Async parallel API calls (N_SAMPLES per question concurrently)
- Incremental saves every 10 questions
- Resume from partial file on restart

Usage:
    python -m src.phase1_baseline                    # run all models
    python -m src.phase1_baseline --model deepseek-v3  # run one model
"""

import json
import asyncio
import argparse
from pathlib import Path

from src.config import (
    MODELS,
    MODEL_ALIASES,
    MODEL_ORDER,
    N_SAMPLES,
    TEMPERATURE,
    MAX_CONCURRENT,
    resolve_model_name,
)
from src.hints import format_base_question, pick_hint_answer
from src.api import get_async_client, async_call_model, extract_answer_letter
from src.utils import compute_question_stats

SAVE_EVERY = 10


async def run_baseline(model_name, model_config, questions):
    """Run all questions without hints, N_SAMPLES times each."""

    print(f"\n{'='*60}")
    print(f"PHASE 1: Baseline for {model_name}")
    print(f"{'='*60}")

    results_dir = Path("results/phase1")
    results_dir.mkdir(parents=True, exist_ok=True)
    partial_path = results_dir / f"baseline_{model_name}_partial.json"
    final_path = results_dir / f"baseline_{model_name}.json"

    # Resume from partial file if it exists
    results = []
    start_idx = 0
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from question {start_idx + 1} ({start_idx} already done)")

    remaining = len(questions) - start_idx
    print(f"Questions: {len(questions)} total, {remaining} remaining")
    print(f"Samples per question: {N_SAMPLES}")
    print(f"Total API calls remaining: {remaining * N_SAMPLES}")
    print()

    client = get_async_client(model_config)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total_tokens = 0

    for q_idx in range(start_idx, len(questions)):
        q = questions[q_idx]
        correct_letter = chr(65 + q['answer'])
        prompt = format_base_question(q)

        # Fire all N_SAMPLES calls concurrently
        tasks = [
            async_call_model(client, model_config, prompt, TEMPERATURE,
                             semaphore)
            for _ in range(N_SAMPLES)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        answers = []
        for s, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"  ERROR on Q{q_idx+1} sample {s+1}: {resp}")
                answers.append({"sample": s, "answer": None, "error": str(resp)})
            else:
                letter = extract_answer_letter(resp["answer_text"])
                total_tokens += resp.get("total_tokens", 0)
                answers.append({
                    "sample": s,
                    "answer": letter,
                    "correct": letter == correct_letter,
                })

        # Compute stats
        stats = compute_question_stats(answers, correct_letter)

        result = {
            "question_idx": q_idx,
            "question": q["question"][:100],
            "subject": q.get("subject", ""),
            "correct_answer": correct_letter,
            "hint_answer": pick_hint_answer(q)[0],
            "answers": answers,
            **stats,
        }
        results.append(result)

        # Print progress
        correct_rate = stats["correct_rate"]
        status = "EASY" if correct_rate >= 0.8 else "HARD" if correct_rate <= 0.2 else "MEDIUM"
        print(f"  Q{q_idx+1:3d}/{len(questions)}: {correct_rate:.0%} correct "
              f"[{status}] dist={stats['answer_distribution']} "
              f"- {q['question'][:40]}...")

        # Incremental save every SAVE_EVERY questions
        if (q_idx + 1) % SAVE_EVERY == 0 or q_idx == len(questions) - 1:
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    hard_questions = [r for r in results if r["correct_rate"] < 0.8]
    print(f"\n--- Summary for {model_name} ---")
    print(f"Total questions: {len(results)}")
    print(f"Hard questions (< 80% correct): {len(hard_questions)}")
    print(f"Total tokens used: {total_tokens:,}")

    # Save final results and clean up partial
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {final_path}")
    if partial_path.exists():
        partial_path.unlink()

    # Save hard question indices for Phase 2
    hard_indices = [r["question_idx"] for r in results if r["correct_rate"] < 0.8]
    hard_path = results_dir / f"hard_questions_{model_name}.json"
    with open(hard_path, "w") as f:
        json.dump(hard_indices, f)
    print(f"Hard question indices saved to {hard_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=sorted(list(MODELS) + list(MODEL_ALIASES)),
                        help="Run only this model (e.g., 'deepseek-v3')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (for quick pilots)")
    args = parser.parse_args()

    # Load questions
    with open("data/questions.json") as f:
        questions = json.load(f)
    if args.limit is not None:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions")

    # Run each model
    if args.model:
        canonical = resolve_model_name(args.model)
        models_to_run = {canonical: MODELS[canonical]}
    else:
        models_to_run = {name: MODELS[name] for name in MODEL_ORDER}

    for model_name, model_config in models_to_run.items():
        asyncio.run(run_baseline(model_name, model_config, questions))


if __name__ == "__main__":
    main()
