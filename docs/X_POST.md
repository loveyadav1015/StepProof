# Draft Post & Thread: StepProof for Meta AI Hackathon

*(Character-optimized draft for X / LinkedIn sharing. No fabricated claims. Focuses on technical architecture, problem-solving, and open weights.)*

---

## 📢 Main Post (Draft)

Manual checklists and PDF manuals don't prevent manufacturing errors when technicians have their hands full.

We built **StepProof** for the #MetaAIHackathon:
📹 Feed it 1 video of an expert doing an assembly.
📋 Meta multimodal VLM extracts an ordered SOP with expected physical end-states.
🎯 SAM 3.1 segments tools and components with zero-shot accuracy.
🚨 Real-time state machine catches skipped steps, wrong parts, or out-of-sequence wiring with visual evidence overlays.

Because the weights can run on-premise, sensitive shop-floor video never has to leave your factory.

Judges & makers can test our offline replay in 30 seconds:
`git clone https://github.com/loveyadav1015/StepProof && make quickstart`

Live demo & code: https://github.com/loveyadav1015/StepProof

#MetaAI #ComputerVision #OpenSource #Manufacturing #SAM3

---

## 🧵 Thread Follow-Ups (Optional)

1/ **The Challenge**: Printed SOPs fail silently. An operator misses a resistor or swaps a red and black wire, and the device fails downstream at quality testing. Traditional computer vision requires months of manual bounding box labeling for every custom circuit.

2/ **The Approach**: StepProof combines Meta's multimodal reasoning with zero-shot spatial grounding. The model writes the verification rules directly from demonstration footage.

3/ **Edge-Ready & Offline**: Industrial floors have limited connectivity and strict data governance. StepProof runs in pure offline cache mode with full cryptographic provenance—tracking token counts, costs, and model hashes down to the frame.

---

## ✅ Viral Track & Sharing Checklist
- [ ] Record a 60-second vertical or 16:9 screen capture GIF of the skipped-step alert triggering with the bounding box overlay.
- [ ] Include the GitHub repository link: `https://github.com/loveyadav1015/StepProof`.
- [ ] Tag `@MetaAI` and include official hackathon hashtags (`#MetaAIHackathon`, `#Llama`, `#SegmentAnything`).
- [ ] Copy the published post URL into `docs/SUBMISSION.md` before final submission cutoff.
