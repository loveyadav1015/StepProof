"""Model pricing and cost calculation based on prices.yaml."""

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


def is_free_tier(model_name: str, prices_path: str | Path = "prices.yaml") -> bool:
    """Check whether a model is marked as free tier in prices.yaml."""
    prices = load_prices(prices_path)
    model_clean = model_name.lower().strip()
    for k, v in prices.items():
        if k != "default" and (k.lower() in model_clean or model_clean in k.lower()):
            if isinstance(v, dict) and v.get("free_tier") is True:
                return True
    return False


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


def calculate_sam_cost(
    model_name: str = "sam-3.1",
    num_images: int = 0,
    num_video_frames: int = 0,
    prices_path: str | Path = "prices.yaml",
) -> float:
    """Calculate SAM segmentation cost in USD per image or video frame."""
    prices = load_prices(prices_path)
    rate = prices.get("sam-3.1", {})
    per_1k_images = float(rate.get("per_1k_images", 2.50))
    per_1k_frames = float(rate.get("per_1k_video_frames", 0.20))

    cost = (num_images / 1000.0 * per_1k_images) + (num_video_frames / 1000.0 * per_1k_frames)
    return round(cost, 6)
