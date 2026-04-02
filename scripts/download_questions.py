"""
download_questions.py — Downloads and merges the experiment dataset.

Creates data/questions.json with 398 questions:
  • 200 MMLU questions — randomly sampled from college/professional-level subjects
  • 198 GPQA Diamond questions — the full Diamond split (hardest GPQA questions,
    validated by domain experts)

Each question has: question, choices, answer (0-indexed), subject, source.
Uses random seed 42 for reproducibility.
"""

from datasets import load_dataset
import json
import random

random.seed(42)


def download_mmlu(n=200):
    """Download and sample MMLU college/professional questions."""
    print("Downloading MMLU...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    print(f"  Total MMLU questions: {len(dataset)}")

    # College and professional-level subjects only
    subjects = [
        "college_mathematics",
        "college_physics",
        "college_chemistry",
        "college_biology",
        "college_computer_science",
        "college_medicine",
        "professional_medicine",
        "professional_accounting",
        "professional_law",
    ]

    filtered = dataset.filter(lambda x: x["subject"] in subjects)
    print(f"  Filtered to {len(filtered)} questions across {len(subjects)} subjects")

    indices = random.sample(range(len(filtered)), min(n, len(filtered)))

    questions = []
    for i in indices:
        row = filtered[i]
        questions.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],  # 0-indexed integer
            "subject": row["subject"],
            "source": "mmlu",
        })

    print(f"  Sampled {len(questions)} MMLU questions")
    return questions


def download_gpqa():
    """Download all 198 GPQA Diamond questions."""
    print("Downloading GPQA Diamond...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print(f"  GPQA Diamond questions: {len(dataset)}")

    # GPQA has separate correct/incorrect answer fields — shuffle into choices
    questions = []
    for row in dataset:
        incorrect = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        choices = incorrect + [row["Correct Answer"]]
        random.shuffle(choices)
        correct_idx = choices.index(row["Correct Answer"])

        questions.append({
            "question": row["Question"],
            "choices": choices,
            "answer": correct_idx,
            "subject": row["Subdomain"],
            "source": "gpqa",
        })

    print(f"  Loaded {len(questions)} GPQA Diamond questions")
    return questions


def main():
    mmlu = download_mmlu(200)
    gpqa = download_gpqa()

    combined = mmlu + gpqa
    print(f"\nTotal: {len(combined)} questions ({len(mmlu)} MMLU + {len(gpqa)} GPQA)")

    with open("data/questions.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("Saved to data/questions.json")

    # Sanity check
    q = combined[0]
    letters = "ABCD"
    print(f"\nSample: [{q['source'].upper()}] {q['subject']}")
    print(f"  {q['question'][:80]}...")
    print(f"  Correct: ({letters[q['answer']]}) {q['choices'][q['answer']][:60]}")


if __name__ == "__main__":
    main()
