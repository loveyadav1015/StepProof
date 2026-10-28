"""Model pricing and token cost calculation based on prices.yaml."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_prices(prices_path: str | Path = "prices.yaml") -> dict[str, Any]:
    """Load pricing definitions from YAML file."""
    path = Path(prices_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load prices from %s: %s", path, e)
        return {}


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    prices_path: str | Path = "prices.yaml",
) -> float:
    """Calculate inference cost in USD from real token usage fields."""
    prices = load_prices(prices_path)
    model_clean = model_name.lower().strip()

    model_key = None
    for k in prices:
        if k != "default" and (k.lower() in model_clean or model_clean in k.lower()):
            model_key = k
            break

    rate = prices.get(model_key, prices.get("default", {"input_per_m": 0.0, "output_per_m": 0.0}))
    input_rate = float(rate.get("input_per_m", 0.0))
    output_rate = float(rate.get("output_per_m", 0.0))

    cost = (input_tokens / 1_000_000.0 * input_rate) + (output_tokens / 1_000_000.0 * output_rate)
    return round(cost, 6)
