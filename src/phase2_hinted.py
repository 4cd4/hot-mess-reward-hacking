"""
phase2_hinted.py — Run hard questions WITH each hint type.

Takes the hard question indices from Phase 1 and runs each one
N_SAMPLES times with each of the 6 hint types. This is the core
data collection for the hint lift and entropy analysis.

Features:
- Async parallel API calls (N_SAMPLES per question concurrently)
- Incremental saves every 10 questions per hint type
- Resume from partial file on restart

Usage:
    python -m src.phase2_hinted                      # run all models
    python -m src.phase2_hinted --model deepseek-v3  # run one model
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
    accepted_model_names,
    resolve_model_name,
)
from src.hints import pick_hint_answer, HINT_TYPES
from src.utils import compute_question_stats
from src.api import get_async_client, async_call_model, extract_answer_letter

SAVE_EVERY = 10


async def run_hinted(model_name, model_config, questions, hard_indices):
    """Run hard questions with all hint types, N_SAMPLES times each."""

    hard_questions = [questions[i] for i in hard_indices]

    print(f"\n{'='*60}")
    print(f"PHASE 2: Hinted runs for {model_name}")
    print(f"{'='*60}")
    print(f"Hard questions: {len(hard_questions)}")
    print(f"Hint types: {len(HINT_TYPES)}")
    print(f"Samples per condition: {N_SAMPLES}")
    print(f"Total API calls: {len(hard_questions) * len(HINT_TYPES) * N_SAMPLES}")
    print()

    results_dir = Path("results/phase2")
    results_dir.mkdir(parents=True, exist_ok=True)
    partial_path = results_dir / f"hinted_{model_name}_partial.json"
    final_path = results_dir / f"hinted_{model_name}.json"

    # Resume from partial file if it exists
    results = {}
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        completed_hints = list(results.keys())
        print(f"Resuming — already completed hints: {completed_hints}")

    client = get_async_client(model_config)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total_tokens = 0

    for hint_name, hint_fn in HINT_TYPES.items():
        # Skip already completed hint types
        if hint_name in results:
            # Check if all questions are done for this hint
            if len(results[hint_name]) >= len(hard_questions):
                print(f"\n--- Skipping {hint_name} (already done) ---")
                continue
            # Partial hint type — resume from where we left off
            start_idx = len(results[hint_name])
            hint_results = results[hint_name]
            print(f"\n--- Resuming hint type: {hint_name} from Q{start_idx+1} ---")
        else:
            start_idx = 0
            hint_results = []
            print(f"\n--- Hint type: {hint_name} ---")

        for q_idx in range(start_idx, len(hard_questions)):
            q = hard_questions[q_idx]
            original_idx = hard_indices[q_idx]
            hint_letter, _ = pick_hint_answer(q)
            correct_letter = chr(65 + q['answer'])
            prompt = hint_fn(q, hint_letter)

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
                    print(f"    ERROR Q{q_idx+1} sample {s+1}: {resp}")
                    answers.append({
                        "sample": s, "answer": None, "error": str(resp)
                    })
                else:
                    letter = extract_answer_letter(resp["answer_text"])
                    total_tokens += resp.get("total_tokens", 0)
                    answers.append({
                        "sample": s,
                        "answer": letter,
                        "followed_hint": letter == hint_letter,
                        "correct": letter == correct_letter,
                    })

            # Compute stats
            stats = compute_question_stats(answers, correct_letter, hint_letter)

            result = {
                "question_idx": original_idx,
                "question": q["question"][:100],
                "correct_answer": correct_letter,
                "hint_answer": hint_letter,
                "answers": answers,
                **stats,
            }

            hint_results.append(result)

            print(f"  Q{q_idx+1:3d}/{len(hard_questions)}: "
                  f"hint={stats.get('hint_follow_rate', 0):.0%} "
                  f"correct={stats['correct_rate']:.0%} "
                  f"({stats['n_valid']} valid) "
                  f"dist={stats['answer_distribution']}")

            # Incremental save every SAVE_EVERY questions
            if (q_idx + 1) % SAVE_EVERY == 0 or q_idx == len(hard_questions) - 1:
                results[hint_name] = hint_results
                with open(partial_path, "w") as f:
                    json.dump(results, f, indent=2)

        # Summary for this hint type
        total_follows = sum(r["hint_follow_rate"] * r["n_valid"]
                          for r in hint_results)
        total_valid = sum(r["n_valid"] for r in hint_results)
        overall_rate = total_follows / max(total_valid, 1)
        print(f"  OVERALL {hint_name}: {overall_rate:.0%} hint follow rate")

        results[hint_name] = hint_results

    print(f"\n--- Cost summary for {model_name} ---")
    print(f"Total tokens: {total_tokens:,}")

    # Save final and clean up partial
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {final_path}")
    if partial_path.exists():
        partial_path.unlink()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=sorted(list(MODELS) + list(MODEL_ALIASES)))
    parser.add_argument("--all", action="store_true",
                        help="Run all questions, not just hard ones")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N (hard) questions (for quick pilots)")
    args = parser.parse_args()

    with open("data/questions.json") as f:
        questions = json.load(f)

    if args.model:
        canonical = resolve_model_name(args.model)
        models_to_run = {canonical: MODELS[canonical]}
    else:
        models_to_run = {name: MODELS[name] for name in MODEL_ORDER}

    for model_name, model_config in models_to_run.items():
        if args.all:
            indices = list(range(len(questions)))
            print(f"Running ALL {len(indices)} questions for {model_name}")
        else:
            hard_path = None
            for candidate in accepted_model_names(model_name):
                candidate_path = Path("results/phase1") / f"hard_questions_{candidate}.json"
                if candidate_path.exists():
                    hard_path = candidate_path
                    break

            if hard_path is None:
                print(f"ERROR: Run phase1_baseline.py for {model_name} first!")
                expected = ", ".join(
                    str(Path("results/phase1") / f"hard_questions_{candidate}.json")
                    for candidate in accepted_model_names(model_name)
                )
                print(f"Missing one of: {expected}")
                continue

            with open(hard_path) as f:
                indices = json.load(f)

            print(f"Loaded {len(indices)} hard question indices for {model_name}")

            if len(indices) == 0:
                print(f"No hard questions found for {model_name}!")
                continue

        if args.limit is not None:
            indices = indices[:args.limit]
            print(f"Limiting to first {len(indices)} questions for {model_name}")

        asyncio.run(run_hinted(model_name, model_config, questions, indices))


if __name__ == "__main__":
    main()
