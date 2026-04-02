"""
config.py — All experiment parameters in one place.
Change these to adjust the experiment scope and cost.
"""

# Random seed for reproducibility (controls question ordering, not API sampling)
SEED = 42

# How many times to sample each question per condition
N_SAMPLES = 5

# Temperature for sampling (need > 0 to get variance)
TEMPERATURE = 1.0

# Max concurrent API requests (semaphore size)
MAX_CONCURRENT = 10


# Canonical model order for scripts that run all models
MODEL_ORDER = ["deepseek-v3", "gpt-4o-mini", "haiku-4.5"]

# Backward-compatible aliases for older result files / CLI usage
MODEL_ALIASES = {}

# Human-readable labels
MODEL_LABELS = {
    "deepseek-v3": "DeepSeek V3",
    "gpt-4o-mini": "GPT-4o-mini",
    "haiku-4.5": "Claude Haiku 4.5",
}


# Models to test
MODELS = {
    "deepseek-v3": {
        "model_id": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "is_reasoning": False,
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": False,
    },
    "haiku-4.5": {
        "model_id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "is_reasoning": False,
    },
}


def resolve_model_name(model_name):
    """Map a CLI/model alias to its canonical model key."""
    canonical = MODEL_ALIASES.get(model_name, model_name)
    if canonical not in MODELS:
        valid = ", ".join(sorted(list(MODELS) + list(MODEL_ALIASES)))
        raise KeyError(f"Unknown model '{model_name}'. Valid options: {valid}")
    return canonical


def accepted_model_names(model_name):
    """Return all known stems for a canonical model, including aliases."""
    canonical = resolve_model_name(model_name)
    names = [canonical]
    names.extend(alias for alias, target in MODEL_ALIASES.items()
                 if target == canonical)
    return names
