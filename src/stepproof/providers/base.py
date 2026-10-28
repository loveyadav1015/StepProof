"""Abstract interfaces for LLM and Segmentation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
import numpy as np

from stepproof.provenance import ProvenanceRecord
from stepproof.schemas import Evidence, SOP, Step

# Image input can be a filesystem Path, raw bytes, or a numpy array (cv2 BGR image)
ImageInput = Union[Path, str, bytes, np.ndarray]


class LLMProvider(ABC):
    """Abstract interface for Multimodal LLM providers (SOP generation & step judgment)."""

    @abstractmethod
    def generate_sop(
        self,
        frames: list[ImageInput],
        prompt: str,
        video_hash: str = "unknown",
    ) -> tuple[SOP, ProvenanceRecord]:
        """Generate a structured SOP from video keyframes and instruction prompt.

        Args:
            frames: Extracted keyframes representing the expert demonstration.
            prompt: Formatted task instruction prompt.
            video_hash: SHA256 hash of the source video.

        Returns:
            Tuple of (validated SOP instance, ProvenanceRecord).
        """
        pass

    @abstractmethod
    def judge_step(
        self,
        frame: ImageInput,
        step: Step,
        evidence: list[Evidence],
        learner_action_desc: str = "",
    ) -> tuple[dict[str, Any], ProvenanceRecord]:
        """Evaluate a learner's execution frame against a step and visual evidence.

        Args:
            frame: Active frame image from learner video.
            step: Step specification with checks and expected end-state.
            evidence: Segmented visual grounding evidence for objects in this frame.
            learner_action_desc: Optional contextual notes.

        Returns:
            Tuple of (verdict dictionary with keys like passed/reason/confidence, ProvenanceRecord).
        """
        pass

    @abstractmethod
    def get_provenance(self) -> ProvenanceRecord:
        """Return the current provider's provenance record."""
        pass


class SegmentationProvider(ABC):
    """Abstract interface for promptable segmentation providers (SAM 3.1 style)."""

    @abstractmethod
    def find(
        self,
        image: ImageInput,
        phrase: str,
        step_id: int | str = 0,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[list[Evidence], ProvenanceRecord]:
        """Find objects in the image corresponding to a single short noun phrase.

        Args:
            image: Image containing candidate objects.
            phrase: Short noun phrase query (e.g. 'red block', 'resistor').
            step_id: Associated step ID for tracking.
            frame_index: Index of the frame in the sampled sequence.
            timestamp_s: Timestamp in seconds.

        Returns:
            Tuple of (list of Evidence objects with bbox/area/confidence, ProvenanceRecord).
        """
        pass

    @abstractmethod
    def get_provenance(self) -> ProvenanceRecord:
        """Return the current provider's provenance record."""
        pass
