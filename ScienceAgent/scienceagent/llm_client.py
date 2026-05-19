"""
Unified LLM client supporting Anthropic and OpenAI-compatible providers.

Provider routing is done by model string prefix:
  "claude-*"            → Anthropic SDK  (ANTHROPIC_API_KEY)
  "openrouter/*"        → OpenRouter API via requests (OPENROUTER_API_KEY)
  "groq/*"              → Groq API, model name passed as-is after stripping "groq/"
                          (OPENAI_API_KEY, base_url=https://api.groq.com/openai/v1)
  "azure/gpt-5.4"      → Azure OpenAI, deployment gpt-5.4-samat
  "azure/gpt-5.4-pro"  → Azure OpenAI, deployment gpt-5.4-pro-samat
  "azure/gpt-5.5"      → Azure OpenAI, deployment gpt-5.5
                          (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
  "gpt-*", "o1-*"      → OpenAI SDK     (OPENAI_API_KEY)
  "openai/*"            → OpenAI SDK with default base_url
  "ollama/*"            → OpenAI-compatible, base_url=http://localhost:11434/v1
  "hf/*"                → HuggingFace Inference, base_url=https://api-inference.huggingface.co/v1
                          (HF_API_KEY)
  "together/*"          → Together.ai, base_url=https://api.together.xyz/v1
                          (TOGETHER_API_KEY). Pass-through model name, e.g.
                          together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
  Any other string      → OpenAI-compatible; set base_url via OPENAI_BASE_URL env var
  Note: ollama seems to be very slow...
"""

import os
import json
import requests
from typing import Optional

# Per-request wall-clock cap for any provider HTTP call. Stops a stalled
# upstream (Together / OpenRouter / HF / etc.) from hanging the whole run.
# 5 minutes accommodates reasoning models (gpt-5.4-pro, DeepSeek-R1, etc.)
# that occasionally need extended thinking time on a fresh world prompt.
HTTP_TIMEOUT_S = 300.0


def complete(
    model: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Send a chat completion request and return the assistant message text.

    Args:
        model: Model identifier string (see module docstring for routing).
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
            Do NOT include a system message here; pass it via `system` instead.
        system: System prompt string (handled natively by Anthropic; prepended
            as a system message for OpenAI-compatible providers).
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's reply as a plain string.

    Note: temperature is intentionally not exposed. Reasoning models
    (gpt-5*, o1*, o3*, claude-opus-4-7+, gpt-5.4-pro) reject it, and
    every provider has a sensible default — passing it just creates
    400-error footguns when adding new model families.
    """
    if _is_anthropic(model):
        return _anthropic_complete(model, messages, system, max_tokens)
    elif model.startswith("openrouter/"):
        return _openrouter_complete(
            model[len("openrouter/") :], messages, system, max_tokens
        )
    elif model.startswith("groq/"):
        return _groq_complete(model[len("groq/") :], messages, system, max_tokens)
    elif model.startswith("azure/"):
        return _azure_complete(model[len("azure/") :], messages, system, max_tokens)
    else:
        return _openai_complete(model, messages, system, max_tokens)


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude")


# ------------------
# anthropic specific


def _anthropic_complete(model, messages, system, max_tokens):
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key, timeout=HTTP_TIMEOUT_S)
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


# ----------
# Openrouter
def _openrouter_complete(model, messages, system, max_tokens):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
        },
        timeout=HTTP_TIMEOUT_S,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
    body = response.json()
    if "choices" not in body:
        raise RuntimeError(f"OpenRouter unexpected response: {body}")
    message = body["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning") or ""
    return content


# ----
# groq
def _groq_complete(model, messages, system, max_tokens):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=HTTP_TIMEOUT_S,
    )

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# -------------
# Azure OpenAI
_AZURE_DEPLOYMENTS = {
    "gpt-5.4": "gpt-5.4-samat",
    "gpt-5.4-pro": "gpt-5.4-pro-samat",
    "gpt-5.5": "gpt-5.5",
}


def _azure_complete(model, messages, system, max_tokens):
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("pip install openai  (>= 1.0)")

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("AZURE_OPENAI_API_KEY not set")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise EnvironmentError("AZURE_OPENAI_ENDPOINT not set")

    deployment = _AZURE_DEPLOYMENTS.get(model)
    if not deployment:
        raise ValueError(
            f"Unknown Azure model '{model}'. " f"Available: {list(_AZURE_DEPLOYMENTS)}"
        )

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2025-04-01-preview",
        timeout=HTTP_TIMEOUT_S,
    )

    # Azure deployments use the Responses API
    kwargs = dict(
        model=deployment,
        input=messages,
        max_output_tokens=max_tokens,
    )
    if system:
        kwargs["instructions"] = system

    response = client.responses.create(**kwargs)
    return response.output_text


# open ai, ollama is not working well, often hangs
_OPENAI_COMPAT_PROVIDERS = {
    "ollama/": ("http://localhost:11434/v1", None),  # no key needed
    "hf/": ("https://api-inference.huggingface.co/v1", "HF_API_KEY"),
    "together/": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openai/": (None, "OPENAI_API_KEY"),  # default OpenAI base
}


def _resolve_openai_provider(model: str) -> tuple[str, Optional[str], Optional[str]]:
    """Return (resolved_model_name, base_url, api_key)."""
    for prefix, (base_url, key_var) in _OPENAI_COMPAT_PROVIDERS.items():
        if model.startswith(prefix):
            resolved = model[len(prefix) :]
            api_key = os.environ.get(key_var) if key_var else "ollama"
            return resolved, base_url, api_key

    # Bare OpenAI model names (gpt-*, o1-*, etc.) or unknown → use OPENAI_API_KEY
    # and optionally OPENAI_BASE_URL for custom endpoints
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    api_key = os.environ.get("OPENAI_API_KEY")
    return model, base_url, api_key


def _openai_complete(model, messages, system, max_tokens):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    resolved_model, base_url, api_key = _resolve_openai_provider(model)

    client_kwargs = {"timeout": HTTP_TIMEOUT_S}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    # OpenAI-compatible providers use the system message inline
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = client.chat.completions.create(
        model=resolved_model,
        messages=full_messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
