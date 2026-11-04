# StepProof — Meta AI Hackathon Official Submission Pack

**Project Title:** StepProof — Industrial Video SOP & Execution Verification  
**Repository URL:** https://github.com/loveyadav1015/StepProof  
**Demonstration Video:** `[INSERT_DEMO_VIDEO_URL_HERE]` (e.g. YouTube / Vimeo / Drive unlisted link)  
**Live Site / Docs:** https://loveyadav1015.github.io/StepProof/ (built with GitHub Pages)  
**Optional Post / Thread:** `[INSERT_OPTIONAL_X_POST_URL_HERE]`  

---

## 1. Project Description (Short & Plain)
Industrial assembly, maintenance, and repair teams rely on paper manuals or static PDF checklists that nobody consults while their hands are busy. Mistakes—such as inserting the wrong component, skipping a resistor, or connecting wires out of order—often escape detection until costly quality control failures occur downstream.

**StepProof** turns a single demonstration video into an active, verifiable Standard Operating Procedure (SOP). Using multimodal AI and zero-shot segmentation, it extracts an ordered sequence of physical steps, grounds the tools and parts in spatial coordinates, and continuously checks a learner’s video against the procedure. A deterministic state machine with multi-frame temporal smoothing flags skipped, out-of-sequence, or incorrect actions in real time and captures auditable visual evidence overlays for supervisors.

---

## 2. Use of Meta Products & Architecture
StepProof architecture combines Meta's multimodal foundation models and computer vision stack:

1. **Multimodal Reasoning (Muse Wonder 1.0 & Spark 1.3)**:
   - **Procedure Decomposition**: Ingests expert footage and generates a structured JSON SOP containing atomic steps, physical object names, expected visual end-states, and common error modes.
   - **Ambiguity Arbiter**: When rule-based spatial segmentation is ambiguous (or when an error is about to be flagged), an LLM judge evaluates the evidence frame to verify whether the visual end-state meets specifications.
2. **Spatial Grounding (SAM 3.1 - Segment Anything Model)**:
   - Provides zero-shot segmentation of components (resistors, jumpers, LEDs, breadboards) directly from natural language phrases.
   - Generates exact bounding boxes and mask area metrics without requiring task-specific training data.
3. **Deterministic Verification State Machine**:
   - Compares real-time learner evidence against SOP preconditions.
   - Enforces multi-frame temporal smoothing to tolerate brief hand occlusions or tool shadows.
   - Disambiguates between out-of-order execution, skipped steps, and wrong part substitution.
4. **Offline & Cryptographic Provenance**:
   - Every inference request maintains a `ProvenanceRecord` recording model IDs, token usage, endpoint latency, and USD cost.
   - The entire system can run in a strictly offline `cache_only` mode (`make quickstart`), allowing instant replay on low-bandwidth shop floors.

---

## 3. Why Meta Models Matter
- **Open Weights & Local Deployment**: Manufacturing shop-floor video often contains proprietary workflows and sensitive intellectual property that organizations cannot send to public third-party APIs. Meta's open-weights models allow the entire perception and reasoning pipeline to execute on-premise on standard workstation hardware.
- **Unified Multimodal Understanding**: Generating accurate assembly procedures requires temporal reasoning across sequential frames that single-frame vision models cannot provide. Meta's multimodal architecture bridges the gap between raw video demonstration and structured industrial logic.

---

## 4. Measured Accuracy & Benchmark Results
*(Backed by repository artifact `eval_report.md` / `eval_report.json`)*:

| Metric | Measured Result | Benchmark Target | Provenance Status |
| :--- | :---: | :---: | :--- |
| **Detection Rate (Staged Errors)** | **100.0%** | 100% | Verified |
| **Correct-Run False Alarms** | **0** | 0 | Verified |
| **Average Latency Per Step** | **< 0.02s** (mock/cached) | < 0.50s | Good |
| **Estimated Cost Per Run** | **$0.0000** | < $0.05 | Verified Low Cost |

*Note on Data Provenance: Benchmark numbers above reflect the synthetic verification suite and dev stand-in replay bundle (`demo_bundle/`). Final live inference runs against Meta developer endpoints (`api.meta.ai`) are configured and ready for live key verification.*

---

## 5. Scope & Safety Boundaries
StepProof is engineered as an **industrial quality-assistance and technician training aid**. It is **not a safety-critical interlock system** and must not replace hardware emergency stops or high-voltage safety interlocks.
