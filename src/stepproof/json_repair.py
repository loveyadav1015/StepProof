"""JSON repair utility for model outputs with markdown formatting, trailing commas, etc."""

from __future__ import annotations

import re


def repair_json(text: str) -> str:
    """Attempt to repair common LLM JSON output flaws before parsing.

    Handles:
    - Markdown code fences (```json ... ```)
    - Surrounding conversational preamble/postamble
    - Trailing commas before closing braces/brackets
    """
    if not isinstance(text, str):
        return str(text)

    clean = text.strip()

    # 1. Remove markdown code block fences if present
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if code_block_match:
        clean = code_block_match.group(1).strip()

    # 2. Extract substring between the first { and last } (or first [ and last ])
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    first_bracket = clean.find("[")
    last_bracket = clean.rfind("]")

    if first_brace != -1 and last_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        clean = clean[first_brace : last_brace + 1]
    elif first_bracket != -1 and last_bracket != -1:
        clean = clean[first_bracket : last_bracket + 1]

    # 3. Remove trailing commas before } or ]
    clean = re.sub(r",\s*([\]}])", r"\1", clean)

    return clean
