"""Mock LLM provider returning deterministic SOPs and step judgments for tests and offline runs."""

from __future__ import annotations

from typing import Any
from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, LLMProvider
from stepproof.schemas import Check, Evidence, SOP, Step


class LLMMockProvider(LLMProvider):
    """Deterministic mock provider that requires zero API keys or external services."""

    def __init__(self, model_name: str = "mock-vision-sop"):
        self.model_name = model_name
        self.provenance = create_provenance(
            provider="mock",
            model_name=self.model_name,
            base_url="http://localhost/mock",
            meta_compliant=False,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def generate_sop(
        self,
        frames: list[ImageInput],
        prompt: str,
        video_hash: str = "synthetic-hash-001",
    ) -> tuple[SOP, ProvenanceRecord]:
        """Return a deterministic 4-step SOP for the synthetic block assembly demonstration."""
        steps = [
            Step(
                id=1,
                instruction="Place the red block in the left assembly slot.",
                objects=["red block"],
                expected_end_state="Red block resting steadily in the left slot.",
                checks=[
                    Check(type="present", object="red block", value=1),
                    Check(type="count", object="red block", value=1),
                ],
                common_errors=["Skipping red block", "Using purple block instead"],
                start_s=0.5,
                end_s=2.0,
            ),
            Step(
                id=2,
                instruction="Place the green block in the center assembly slot.",
                objects=["green block"],
                expected_end_state="Green block positioned directly adjacent to red block in center slot.",
                checks=[
                    Check(type="present", object="green block", value=1),
                    Check(type="count", object="green block", value=1),
                ],
                common_errors=["Skipping green block", "Placing blue block first"],
                start_s=2.0,
                end_s=4.0,
            ),
            Step(
                id=3,
                instruction="Place the blue block in the right assembly slot.",
                objects=["blue block"],
                expected_end_state="Blue block resting in right slot.",
                checks=[
                    Check(type="present", object="blue block", value=1),
                    Check(type="count", object="blue block", value=1),
                ],
                common_errors=["Skipping blue block", "Placing out of sequence"],
                start_s=4.0,
                end_s=6.0,
            ),
            Step(
                id=4,
                instruction="Place the yellow block on top as the lock piece.",
                objects=["yellow block"],
                expected_end_state="Yellow block placed on top forming completed assembly.",
                checks=[
                    Check(type="present", object="yellow block", value=1),
                    Check(type="count", object="yellow block", value=1),
                ],
                common_errors=["Skipping lock piece"],
                start_s=6.0,
                end_s=8.0,
            ),
        ]

        sop = SOP(
            task="Colored Block Assembly",
            steps=steps,
            source_video_hash=video_hash,
            created_with=self.provenance,
        )
        return sop, self.provenance

    def judge_step(
        self,
        frame: ImageInput,
        step: Step,
        evidence: list[Evidence],
        learner_action_desc: str = "",
    ) -> tuple[dict[str, Any], ProvenanceRecord]:
        """Judge whether step end-state is satisfied based on provided visual evidence."""
        # Check if all required objects have positive evidence with mask area > 0
        required_objects = {obj.lower().strip() for obj in step.objects}
        found_objects = {
            ev.object.lower().strip()
            for ev in evidence
            if ev.mask_area_px > 0 and ev.confidence >= 0.5
        }

        missing = required_objects - found_objects
        if not missing:
            verdict = {
                "passed": True,
                "reason": f"Step {step.id} verified: all objects ({', '.join(step.objects)}) located in expected state.",
                "confidence": 0.96,
                "failed_rules": [],
            }
        else:
            verdict = {
                "passed": False,
                "reason": f"Step {step.id} incomplete: missing required object(s): {', '.join(missing)}.",
                "confidence": 0.92,
                "failed_rules": [f"Object '{obj}' not detected in expected state" for obj in missing],
            }

        return verdict, self.provenance
