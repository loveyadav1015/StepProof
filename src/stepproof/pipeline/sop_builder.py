"""Build structured SOP from expert demonstration frames using multimodal LLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from stepproof.cache import SQLiteCache
from stepproof.config import settings
from stepproof.ingest.sampler import SampledVideo
from stepproof.provenance import ProvenanceRecord
from stepproof.providers.base import LLMProvider
from stepproof.schemas import SOP

logger = logging.getLogger(__name__)

SOP_PROMPT_VERSION = "v1.0"

SOP_SYSTEM_PROMPT = """You are an industrial engineer specializing in manual assembly Standard Operating Procedures (SOPs).
Analyze the provided sequential keyframes of an expert demonstration.
Extract an ordered sequence of discrete steps.

For each step:
1. id: Sequential integer (1, 2, ...).
2. instruction: Clear, imperative command (e.g. "Place the red block in the left slot.").
3. objects: Array of physical objects manipulated (e.g. ["red block"]).
4. expected_end_state: Observable visual condition once the step is completed.
5. checks: Array of verification rules. Each check must specify:
   - type: "present" | "count" | "order" | "position"
   - object: exact noun phrase matching the object
   - value: expected count (e.g. 1) or target coordinate/region
6. common_errors: Array of potential learner mistakes (skipped step, wrong part, out of sequence).
7. start_s: Approximate start time in seconds.
8. end_s: Approximate end time in seconds.

Return JSON conforming strictly to the SOP schema:
{
  "task": "Task name",
  "steps": [ ... ]
}
"""


def build_sop(
    sampled_video: SampledVideo,
    llm_provider: LLMProvider,
    output_path: Path | str | None = None,
    cache: SQLiteCache | None = None,
    mode: str | None = None,
) -> tuple[SOP, ProvenanceRecord]:
    """Extract and validate SOP from expert video frames with caching.

    Args:
        sampled_video: Video sampling metadata with keyframe paths.
        llm_provider: LLM provider instance (e.g. LLMMockProvider, LLMOpenAICompatProvider).
        output_path: Optional path to write output sop.json.
        cache: Optional SQLite cache instance.
        mode: Cache mode ('live', 'cache_only', 'refresh').

    Returns:
        Tuple of (validated SOP, ProvenanceRecord).
    """
    frame_paths = [f.file_path for f in sampled_video.frames]
    video_hash = sampled_video.video_hash
    prov = llm_provider.get_provenance()

    def _call_model() -> tuple[dict, ProvenanceRecord]:
        sop_obj, prov_record = llm_provider.generate_sop(
            frames=frame_paths,
            prompt=SOP_SYSTEM_PROMPT,
            video_hash=video_hash,
        )
        return sop_obj.model_dump(), prov_record

    if cache is not None:
        raw_data, provenance = cache.execute_with_cache(
            provider_name=prov.provider,
            model_name=prov.model_name,
            prompt_version=SOP_PROMPT_VERSION,
            input_data={"video_hash": video_hash, "prompt": SOP_PROMPT_VERSION},
            caller_fn=_call_model,
            mode=mode or settings.stepproof_mode,
        )
        sop = SOP.model_validate(raw_data)
    else:
        raw_data, provenance = _call_model()
        sop = SOP.model_validate(raw_data)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(sop.model_dump_json(indent=2))
        logger.info("Saved SOP to %s", out)

    return sop, provenance
