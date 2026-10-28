# StepProof Architecture

StepProof is an offline-first video verification engine designed for the **Meta AI Developer Hackathon**. It turns an expert video of a manual task into a structured Standard Operating Procedure (SOP) and checks a learner's execution video, identifying skipped, wrong, or out-of-order steps with visual evidence.

---

## 1. System Overview

StepProof operates across five core stages:

```
[Expert Video] ──> Ingest/Sampler ──> LLM Provider (SOP Builder) ──> Golden SOP (JSON)
                                                                           │
                                                                           ▼
[Learner Video] ──> Ingest/Sampler ──> Segmentation & Grounding ──> State Machine Checker
                                                                           │
                                                                           ▼
                                                               Verification Report & Streamlit UI
```

1. **Ingest & Change Detection**: Extracts timestamped frames (~2 fps) and detects scene transitions with OpenCV. Enforces `ffmpeg` system presence.
2. **Understanding (SOP Builder)**: Uses vision LLM (e.g., Muse Glimmer / Wonder or OpenAI-compatible endpoint) to extract ordered task steps, object lists, expected end states, and validation rules (`Check`).
3. **Grounding**: Prompts SAM 3.1 / Segmentation provider with short noun phrases per step to locate objects, measure area, and compute reference bounding boxes.
4. **Checking & State Machine**: Evaluates learner frames against SOP rules. A step transitions `PENDING` ➔ `ACTIVE` ➔ `DONE` when all rules hold over 3 consecutive sampled frames. Flags skipped, wrong, and out-of-order steps.
5. **Presentation**: Streamlit interface and Markdown reports showing step checklists, visual bounding boxes/mask overlays, and provenance records.

---

## 2. Meta Compliance & Provenance Tracking

The hackathon requires incorporating Meta models (SAM 3.1, Muse Wonder, Muse Spark, Muse Glimmer).

To prevent unintended disqualification or silent fallback to non-Meta models:
- Every model output (cached items, SOPs, step verdicts, reports) carries a `ProvenanceRecord`:
  - `provider`: provider identifier (`mock`, `openai_compat`, `local_sam`, `meta_api`)
  - `model_name`: exact model name string
  - `base_url_host`: host domain or IP (e.g. `api.meta.ai`, `openrouter.ai`, `localhost`)
  - `timestamp`: UTC ISO-8601 timestamp
  - `meta_compliant`: boolean flag. True **only** for official Meta models:
    - SAM 3.1 (`facebookresearch/sam3.1` or Meta Model API `sam-3.1`)
    - Muse Glimmer 30B, Muse Spark, Muse Wonder
- **`STEPPROOF_REQUIRE_META=1` Guard**:
  When active:
  - The provider factory refuses instantiation of any provider that produces non-meta-compliant results.
  - The cache and UI refuse to load records where `meta_compliant == False`.
  - Fail-fast guarantee: **Zero silent fallbacks** from Meta to non-Meta models.

---

## 3. Provider Abstraction Layer

All model operations sit behind abstract interfaces in `src/stepproof/providers/base.py`:

### `LLMProvider`
- `generate_sop(frames, prompt) -> tuple[SOP, ProvenanceRecord]`
- `judge_step(frame, step, evidence) -> tuple[dict, ProvenanceRecord]`

Implementations:
- `LLMMockProvider`: Deterministic outputs for offline development, synthetic videos, and unit tests.
- `LLMOpenAICompatProvider`: Generic client supporting Ollama (`http://localhost:11434/v1`), OpenRouter, Groq, and the official Meta Model API (`https://api.meta.ai/v1`). Supports JSON mode, base64 image frames, optional temperature, and optional reasoning effort.

### `SegmentationProvider`
- `find(image, phrase) -> tuple[list[Evidence], ProvenanceRecord]`

Implementations:
- `SegMockProvider`: OpenCV contour and color-based object detector calibrated on synthetic block-assembly footage. Zero API keys or GPU weights needed.
- `SegLocalSAMProvider`: Stub with explicit instructions for `facebookresearch/sam3` weights. Raises `NotImplementedError`.
- `SegMetaAPIProvider`: Stub for the Meta Model API Media Segmentation endpoint (`https://dev.meta.ai/docs/overview`). Raises `NotImplementedError`.

---

## 4. Cache System (`cache.py`)

StepProof uses an embedded SQLite database to cache heavy model inferences:
- **Key formula**: `sha256(provider + ":" + model + ":" + prompt_version + ":" + input_hash)`
- **Execution Modes (`STEPPROOF_MODE`)**:
  - `live`: Checks cache; on miss, queries provider, stores result + provenance, and returns.
  - `cache_only`: Replays exclusively from cache. Raises `KeyError` on miss. Perfect for live demos without internet risk.
  - `refresh`: Executes live call and overwrites existing cache entries.

---

## 5. Checker State Machine Logic

The checker (`src/stepproof/pipeline/checker.py`) processes sampled frames sequentially:

```
[PENDING] ────(evidence detected)────> [ACTIVE] ────(rules pass 3 frames)────> [DONE]
    │                                     │
    │ (step n+1 satisfied before n)       │ (window elapsed / wrong part)
    ▼                                     ▼
[FLAGGED: skipped]                   [FLAGGED: wrong]
```

- **DONE Condition**: All checks (`present`, `count`, `position`, `order`) pass for **3 consecutive sampled frames**.
- **Skipped Step**: If step $k > n$ has its end-state satisfied while step $n$ is still `PENDING` or `ACTIVE`, step $n$ is flagged as `skipped`.
- **Wrong Part / Placement**: If a rule fails at the end of a step's sampling window or when the learner video concludes without satisfying the rule, step is flagged as `wrong`.
- **Out-of-Order**: If a later step completes before an earlier step, earlier or out-of-order steps are flagged.
- **Merge Resolution**:
  - Hard segmentation rule failure ➔ `FLAGGED`.
  - Segmentation rules pass, but LLM judge expresses high-confidence failure ➔ `needs_review`.
  - Both pass ➔ `DONE`.

---

## 6. Directory Structure

```
stepproof/
├── pyproject.toml
├── README.md
├── Makefile
├── .env.example
├── .gitignore
├── artifacts/
│   └── Research.md
├── docs/
│   └── ARCHITECTURE.md
├── src/stepproof/
│   ├── __init__.py
│   ├── __main__.py               # CLI entrypoint
│   ├── config.py                 # Pydantic / env settings
│   ├── schemas.py                # Pydantic schemas (SOP, Check, Evidence, Verdict, Report)
│   ├── provenance.py             # ProvenanceRecord & Meta compliance checks
│   ├── cache.py                  # SQLite caching layer
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── sampler.py            # OpenCV frame extraction & scene change detection
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract provider interfaces
│   │   ├── factory.py            # Provider factory with Meta compliance guard
│   │   ├── llm_openai_compat.py  # Generic OpenAI client (Ollama/OpenRouter/Meta API)
│   │   ├── llm_mock.py           # Deterministic offline mock LLM
│   │   ├── seg_mock.py           # OpenCV color/contour segmentation mock
│   │   ├── seg_local_sam.py      # Stub for local SAM 3.1
│   │   └── seg_meta_api.py       # Stub for Meta API SAM
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── sop_builder.py        # SOP prompt construction and schema validation
│   │   ├── grounding.py          # Object grounding and mask overlay generation
│   │   ├── checker.py            # Rule evaluation and step state machine
│   │   └── report.py             # Markdown and JSON report generator
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py      # 3-tab Streamlit dashboard
├── scripts/
│   └── make_synthetic_video.py   # Synthetic video test dataset generator
└── tests/
    ├── __init__.py
    ├── test_schemas.py
    ├── test_cache.py
    ├── test_providers.py
    ├── test_checker.py
    ├── test_sampler.py
    └── test_cli.py
```
