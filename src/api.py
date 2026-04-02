"""
api.py — Handles calling different model APIs.
Wraps OpenAI-compatible APIs (DeepSeek, OpenAI) and Anthropic into one interface.
Supports both sync and async clients.
"""

import os
import re
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import anthropic

load_dotenv()


def get_client(model_config):
    """Create a sync API client for the given model config."""
    api_key = os.environ.get(model_config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"Missing API key: set {model_config['api_key_env']} in .env"
        )
    if model_config.get("provider") == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    return OpenAI(api_key=api_key, base_url=model_config["base_url"])


def get_async_client(model_config):
    """Create an async API client for the given model config."""
    api_key = os.environ.get(model_config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"Missing API key: set {model_config['api_key_env']} in .env"
        )
    if model_config.get("provider") == "anthropic":
        return anthropic.AsyncAnthropic(api_key=api_key)
    return AsyncOpenAI(api_key=api_key, base_url=model_config["base_url"])


def _build_kwargs(model_config, prompt, temperature):
    """Build the kwargs dict for a chat completion call."""
    if model_config.get("provider") == "anthropic":
        return {
            "model": model_config["model_id"],
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
    kwargs = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
    }
    kwargs["temperature"] = temperature
    return kwargs


def _parse_response(response):
    """Extract answer text, tokens, and reasoning from a response."""
    answer_text = response.choices[0].message.content or ""
    result = {
        "answer_text": answer_text,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return result


def _parse_anthropic_response(response):
    """Extract answer text and tokens from an Anthropic response."""
    answer_text = ""
    for block in response.content:
        if block.type == "text":
            answer_text += block.text
    return {
        "answer_text": answer_text,
        "total_tokens": (response.usage.input_tokens + response.usage.output_tokens)
                        if response.usage else 0,
    }


def call_model(client, model_config, prompt, temperature):
    """
    Send a prompt to a model and return the response text (sync).
    Returns a dict with 'answer_text' and optionally 'reasoning'.
    """
    kwargs = _build_kwargs(model_config, prompt, temperature)
    if model_config.get("provider") == "anthropic":
        response = client.messages.create(**kwargs)
        return _parse_anthropic_response(response)
    response = client.chat.completions.create(**kwargs)
    return _parse_response(response)


async def async_call_model(client, model_config, prompt, temperature,
                           semaphore):
    """
    Send a prompt to a model and return the response text (async).
    Uses semaphore to limit concurrency. Retries on rate limit errors.
    """
    is_anthropic = model_config.get("provider") == "anthropic"
    max_retries = 5
    delay = 0.5
    for attempt in range(max_retries):
        try:
            async with semaphore:
                kwargs = _build_kwargs(model_config, prompt, temperature)
                if is_anthropic:
                    response = await client.messages.create(**kwargs)
                    return _parse_anthropic_response(response)
                else:
                    response = await client.chat.completions.create(**kwargs)
                    return _parse_response(response)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 32)
                    continue
            else:
                print(f"  API error (attempt {attempt+1}): {error_str[:100]}")
            raise


def extract_answer_letter(text):
    """
    Extract the answer letter (A/B/C/D) from model response text.
    Looks for ANSWER: X first (our requested format), then fallbacks.
    """
    if not text:
        return None

    t = text.strip()
    upper = t.upper()

    # Prefer explicit "final answer" formats, taking the *last* match.
    primary_patterns = [
        r"\bANSWER\s*:\s*\(?\s*([A-D])\s*\)?\b",
        r"\bFINAL\s+ANSWER\s*:\s*\(?\s*([A-D])\s*\)?\b",
        r"\bFINAL\s+ANSWER\s+IS\s*\(?\s*([A-D])\s*\)?\b",
        r"\bTHE\s+ANSWER\s+IS\s*\(?\s*([A-D])\s*\)?\b",
        r"\bTHE\s+CORRECT\s+ANSWER\s+IS\s*\(?\s*([A-D])\s*\)?\b",
    ]
    for pattern in primary_patterns:
        matches = re.findall(pattern, upper)
        if matches:
            return matches[-1]

    # Secondary: letter at end of response (common for terse outputs).
    tail_patterns = [
        r"\b([A-D])\b\s*$",
        r"\(([A-D])\)\s*$",
    ]
    for pattern in tail_patterns:
        m = re.search(pattern, upper)
        if m:
            return m.group(1)

    # Last resort: scan last few lines for a standalone A/B/C/D token.
    lines = [ln.strip() for ln in upper.splitlines() if ln.strip()]
    for ln in reversed(lines[-5:]):
        m = re.search(r"^\(?\s*([A-D])\s*\)?\.?$", ln)
        if m:
            return m.group(1)

    return None
