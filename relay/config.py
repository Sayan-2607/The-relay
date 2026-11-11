"""Central config. Everything here is overridable by env vars or limits.json."""
import json, os
from dataclasses import dataclass

FAST_MODEL = os.environ.get("RELAY_FAST_MODEL", "openai/gpt-oss-20b")
STRONG_MODEL = os.environ.get("RELAY_STRONG_MODEL", "openai/gpt-oss-120b")

# USD per 1M tokens (in, out). CHECK https://groq.com/pricing - these are my recollection, not verified.
# The free tier costs $0; these give a *list-price equivalent* so "cost per 1,000 answers" means something.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}


@dataclass
class Endpoint:
    name: str
    base_url: str
    api_key: str


def endpoint(source: str) -> Endpoint:
    if source == "real":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY in your environment (never commit it).")
        return Endpoint("real", os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"), key)
    return Endpoint("standin", os.environ.get("STANDIN_URL", "http://127.0.0.1:8099/v1"), "standin-key")


def load_limits(path="limits.json"):
    """Per-model limits the *governor* believes. Replace with what probe.py found (Table 1).
    Defaults are my recollection of Groq free-tier numbers: UNVERIFIED placeholders."""
    from .governor import Limits
    defaults = {
        FAST_MODEL: dict(rpm=30, tpm=6000, rpd=14400, tpd=500000, max_inflight=4),
        STRONG_MODEL: dict(rpm=30, tpm=12000, rpd=1000, tpd=100000, max_inflight=4),
    }
    if os.path.exists(path):
        with open(path) as f:
            defaults.update(json.load(f))
    return {m: Limits(**v) for m, v in defaults.items()}
