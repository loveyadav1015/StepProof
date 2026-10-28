"""Pydantic schemas for SOP, Checks, Evidence, Verdicts, and Run Reports."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, ValidationError

from stepproof.provenance import ProvenanceRecord

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

CheckType = Literal["present", "count", "order", "position"]
StepState = Literal["PENDING", "ACTIVE", "DONE", "FLAGGED"]
FlagType = Literal["skipped", "wrong", "out_of_order", "needs_review", "none"]


class Check(BaseModel):
    """Validation rule for a single step end-state."""

    type: CheckType = Field(description="Check category: present, count, order, position")
    object: str = Field(description="Noun phrase or object target for check")
    value: Any = Field(default=None, description="Expected value (e.g. count=1, position coords)")


class Step(BaseModel):
    """A discrete task step extracted from an expert demonstration."""

    id: int | str = Field(description="Unique step identifier (1-indexed or string tag)")
    instruction: str = Field(description="Imperative instruction describing the action")
    objects: list[str] = Field(default_factory=list, description="List of physical objects involved")
    expected_end_state: str = Field(description="Observable visual condition after step completion")
    checks: list[Check] = Field(default_factory=list, description="Rules to verify end-state")
    common_errors: list[str] = Field(default_factory=list, description="Common mistakes learner might make")
    start_s: float = Field(default=0.0, description="Approximate start timestamp in expert video")
    end_s: float = Field(default=0.0, description="Approximate end timestamp in expert video")


class SOP(BaseModel):
    """Standard Operating Procedure generated from expert video."""

    task: str = Field(description="Overall task name or goal")
    steps: list[Step] = Field(default_factory=list, description="Ordered sequence of steps")
    source_video_hash: str = Field(description="SHA256 hash of the expert demonstration video")
    created_with: ProvenanceRecord = Field(description="Model provenance record for SOP creation")


class Evidence(BaseModel):
    """Visual grounding evidence found in a video frame."""

    step_id: int | str = Field(description="Associated step identifier")
    frame_index: int = Field(description="Sampled frame index")
    timestamp_s: float = Field(description="Frame timestamp in seconds")
    object: str = Field(description="Object name or label detected")
    count: int = Field(default=1, description="Detected instance count")
    mask_area_px: int = Field(default=0, description="Area of segmentation mask in pixels")
    bbox: list[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Bounding box [x, y, w, h] in pixels",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0.0-1.0")


class StepVerdict(BaseModel):
    """Verification verdict for a single step in a learner run."""

    step_id: int | str = Field(description="Target step identifier")
    state: StepState = Field(default="PENDING", description="Final state: PENDING, ACTIVE, DONE, FLAGGED")
    reason: str = Field(default="", description="Human-readable explanation of verdict")
    failed_rules: list[str] = Field(default_factory=list, description="List of rule descriptions that failed")
    evidence_frame_path: Optional[str] = Field(
        default=None, description="Relative path to key evidence frame image"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of judgment")
    flag_type: FlagType = Field(
        default="none",
        description="Type of failure flagged: skipped, wrong, out_of_order, needs_review, none",
    )
    secondary_flag_type: Optional[FlagType] = Field(
        default=None,
        description="Optional secondary failure flag type when multiple conditions apply",
    )


class RunReport(BaseModel):
    """Complete execution summary of a verification run."""

    sop_id: str = Field(description="Task or SOP identifier")
    verdicts: list[StepVerdict] = Field(default_factory=list, description="Verdicts for each step")
    timings: dict[str, float] = Field(
        default_factory=dict, description="Execution timings for pipeline stages (seconds)"
    )
    provenance: list[ProvenanceRecord] = Field(
        default_factory=list, description="Provenance records for all models invoked in this run"
    )
    cost_estimate: dict[str, float] | float = Field(
        default_factory=dict, description="Estimated API or compute cost in USD"
    )


def validate_with_retry(
    model_cls: type[T],
    data_provider: Callable[[], Any],
    max_retries: int = 1,
) -> T:
    """Validate data against a Pydantic model with retry on failure.

    Args:
        model_cls: Target Pydantic model class.
        data_provider: Callable returning raw dict or JSON data to validate.
        max_retries: Maximum number of retry attempts before raising.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValidationError: If validation fails after all retry attempts.
    """
    attempts = 0
    last_error: Optional[ValidationError] = None

    while attempts <= max_retries:
        raw_data = data_provider()
        try:
            if isinstance(raw_data, model_cls):
                return raw_data
            if isinstance(raw_data, dict):
                return model_cls.model_validate(raw_data)
            if isinstance(raw_data, str):
                return model_cls.model_validate_json(raw_data)
            return model_cls.model_validate(raw_data)
        except ValidationError as e:
            attempts += 1
            last_error = e
            logger.warning(
                "Validation failed for %s on attempt %d/%d: %s",
                model_cls.__name__,
                attempts,
                max_retries + 1,
                e,
            )
            if attempts > max_retries:
                raise e

    raise last_error or RuntimeError(f"Failed to validate {model_cls.__name__}")
