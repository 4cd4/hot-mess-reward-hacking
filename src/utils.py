"""
utils.py — Shared utility functions for stats computation.
Single source of truth to avoid duplication across phase/patch scripts.
"""


def compute_question_stats(answers, correct_letter, hint_letter=None):
    """
    Compute per-question statistics from a list of answer dicts.
    Each answer dict must have at least {"answer": "A"|"B"|"C"|"D"|None}.

    Returns a dict with:
        answer_distribution: {"A": int, "B": int, "C": int, "D": int}
        correct_rate: float
        hint_follow_rate: float (only if hint_letter provided)
        n_valid: int
    """
    valid = [a for a in answers if a.get("answer")]
    n_valid = len(valid)

    # Always include all 4 letters for consistency
    dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for a in valid:
        letter = a["answer"]
        if letter in dist:
            dist[letter] += 1

    correct_count = sum(1 for a in valid if a["answer"] == correct_letter)
    correct_rate = correct_count / max(n_valid, 1)

    stats = {
        "answer_distribution": dist,
        "correct_rate": correct_rate,
        "n_valid": n_valid,
    }

    if hint_letter is not None:
        hint_count = sum(1 for a in valid if a["answer"] == hint_letter)
        stats["hint_follow_rate"] = hint_count / max(n_valid, 1)

    return stats
