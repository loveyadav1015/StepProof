# StepProof 👁️

> **Offline-first video-to-SOP generation and learner step verification engine with visual evidence.**  
> Built for the **Meta AI Developer Hackathon 2026**.

---

## 📖 What is StepProof?

StepProof turns a single **expert video demonstration** of a manual task into a structured Standard Operating Procedure (**SOP**) with discrete steps, target objects, expected visual end-states, and validation rules. It then analyzes a **learner's execution video**, validating physical progress and flagging **skipped**, **wrong/failed**, or **out-of-order** steps with annotated keyframe evidence.

StepProof is designed around a **free-first, Meta-compliant architecture**:
1. **Develop for Free**: Run offline with OpenCV mock segmenters and deterministic mock LLMs or local vision models (Ollama / Groq) with **zero API keys and zero cost**.
2. **Swap to Meta Models for Submission**: Seamlessly plug in **SAM 3.1** (local weights or Meta Model API) and **Muse Glimmer / Muse Wonder / Muse Spark** behind swappable provider interfaces.
3. **Meta Compliance Enforcement**: A strict compliance guard (`STEPPROOF_REQUIRE_META=1`) guarantees that all model inferences, cached results, and generated reports originate from official Meta models—preventing accidental silent fallback to non-Meta models.

---

## ⚡ Prerequisites

- **Python 3.11+**
- **ffmpeg** (system executable): StepProof uses `ffmpeg` for video decoding, downsampling, and keyframe extraction.
  ```bash
  # Linux (Ubuntu / Debian)
  sudo apt update && sudo apt install -y ffmpeg

  # macOS
  brew install ffmpeg
  ```
  *Note:* StepProof asserts `ffmpeg` presence at startup and raises a clear runtime error if it is missing from `PATH`.

---

## 🚀 Quick Start (With Zero API Keys)

StepProof is ready to run immediately using synthetic demo footage and offline mock providers.

### 1. Install & Build Environment
```bash
make install
```
This initializes a Python 3.11 virtual environment (`.venv`), upgrades `pip`, and installs all dependencies in editable mode.

### 2. Run Test Suite
```bash
make test
```
Runs 22 unit and smoke tests covering schema validation, SQLite cache modes, rule evaluation state machines, provider factory, and CLI execution.

### 3. Run Verification Demo
```bash
make demo
```
This single command:
1. Generates synthetic expert and learner demonstration videos (`expert.mp4`, `learner_correct.mp4`, `learner_skipped.mp4`, `learner_wrong_part.mp4`, `learner_out_of_order.mp4`).
2. Extracts and saves a golden SOP (`data/outputs/demo_sop.json`).
3. Grounds physical objects using the color/contour segmenter and generates reference overlays (`data/outputs/grounding/`).
4. Runs verification across all learner test videos, catching and flagging the staged errors.
5. Produces an audit report at `data/outputs/runs/skipped/report.md`.

### 4. Launch Streamlit UI
```bash
make app
```
Opens the interactive 3-tab dashboard at `http://localhost:8501`:
- **Tab 1 (Build SOP)**: Select expert video, trigger frame extraction, inspect golden SOP checklist, and generate SAM grounding overlays.
- **Tab 2 (Check Run)**: Select learner video (correct, skipped, wrong part, out-of-order), trigger verification, view live status ticking (`DONE` vs `FLAGGED`), and inspect evidence frame callouts.
- **Tab 3 (Report & Audit)**: Browse previous runs, inspect model provenance badges, and download `report.md` / `report.json`.

---

## 🔄 Switching Providers & Stand-ins

All model integrations sit behind abstract interfaces (`LLMProvider`, `SegmentationProvider`) and are configured strictly through environment variables. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### Free Dev Stand-in: Local Ollama (Ollama Vision)
Run local multimodal models without spending credits:
```bash
# In .env
STEPPROOF_LLM=openai_compat
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2-vision
LLM_API_KEY=ollama
LLM_API_STYLE=chat
STEPPROOF_REQUIRE_META=0
```

### Free Dev Stand-in: Groq
```bash
# In .env
STEPPROOF_LLM=openai_compat
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.2-11b-vision-preview
LLM_API_KEY=gsk_your_groq_api_key
STEPPROOF_REQUIRE_META=0
```

### Near-Free Fallback: OpenRouter Muse Glimmer 30B
Run Meta's distilled open agentic model on OpenRouter:
```bash
# In .env
STEPPROOF_LLM=openai_compat
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta/muse-glimmer-30b
LLM_API_KEY=sk-or-v1-your-key
STEPPROOF_REQUIRE_META=0  # Set to 1 if OpenRouter open-weight run is confirmed compliant
```

---

## 🏆 Meta Hackathon Submission & Compliance

> [!IMPORTANT]
> **The final hackathon submission MUST run with `STEPPROOF_REQUIRE_META=1`**.

### The Meta Compliance Guard
When `STEPPROOF_REQUIRE_META=1`:
1. The **Provider Factory** refuses to instantiate any provider marked `meta_compliant=False` (rejecting mocks, Groq, Ollama dev models, etc.).
2. The **Cache System** refuses to load any cached inferences whose recorded provenance is not Meta compliant.
3. The **Streamlit App** and **CLI** display explicit compliance badges (`🟢 Meta model` vs `⚪ Dev stand-in`) on all step verdicts and markdown reports.
4. **Zero Silent Fallback**: The pipeline will immediately fail fast with an informative error rather than silently falling back to a non-Meta model.

### Swapping to Meta Models for Submission

#### 1. Official Meta Model API (Muse Wonder / Spark)
```bash
# In .env
STEPPROOF_LLM=openai_compat
LLM_BASE_URL=https://api.meta.ai/v1
LLM_MODEL=muse-wonder
LLM_API_KEY=your_meta_model_api_key
LLM_API_STYLE=chat
STEPPROOF_REQUIRE_META=1
```

#### 2. SAM 3.1 Local Weights
```bash
# In .env
STEPPROOF_SEG=local_sam
STEPPROOF_REQUIRE_META=1
```
Follow the setup instructions in `src/stepproof/providers/seg_local_sam.py`:
1. Request access at `https://huggingface.co/facebook/sam3.1` or `https://github.com/facebookresearch/sam3`.
2. Place approved weights in `weights/sam3.1/` (gitignored).
3. Connect the predictor call in `SegLocalSAMProvider.find()`.

#### 3. SAM 3.1 Meta Model API
```bash
# In .env
STEPPROOF_SEG=meta_api
STEPPROOF_REQUIRE_META=1
```
Consult the Media segmentation documentation at `https://dev.meta.ai/docs/overview` and fill in `SegMetaAPIProvider.find()`.

---

## 🛠️ CLI Usage

```bash
# Generate SOP from expert footage
python -m stepproof build-sop data/expert/expert.mp4 --out data/outputs/sop.json

# Ground physical objects on expert keyframes
python -m stepproof ground data/expert/expert.mp4 --sop data/outputs/sop.json --out-dir data/outputs/grounding

# Check learner demonstration video
python -m stepproof check data/learner/learner_skipped.mp4 --sop data/outputs/sop.json --out-dir data/outputs/runs/my_run

# Display or export Markdown report
python -m stepproof report data/outputs/runs/my_run --out data/outputs/runs/my_run/report.md

# Replay cached run with compliance check
python -m stepproof replay data/outputs/runs/my_run --require-meta
```

---

## 📁 Repository Layout

```
stepproof/
├── pyproject.toml              # Project configuration and dependencies
├── Makefile                    # Targets: install, test, lint, app, demo
├── .env.example                # Sample environment configurations (Ollama, Meta API)
├── .gitignore                  # Excludes .env, data/, media, caches, weights
├── artifacts/
│   └── Research.md             # Background hackathon research & architecture plan
├── docs/
│   └── ARCHITECTURE.md         # Full architectural design and workflow details
├── src/stepproof/
│   ├── __init__.py
│   ├── __main__.py             # CLI entrypoint (build-sop, ground, check, report, replay)
│   ├── config.py               # Env-driven settings
│   ├── schemas.py              # Pydantic models (SOP, Check, Evidence, Verdict, Report)
│   ├── provenance.py           # ProvenanceRecord tracking & Meta compliance guards
│   ├── cache.py                # SQLite response cache (live, cache_only, refresh)
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── sampler.py          # OpenCV frame extraction (~2 fps) + scene-change detection
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py             # LLMProvider and SegmentationProvider interfaces
│   │   ├── factory.py          # Provider factory with require-meta enforcement
│   │   ├── llm_openai_compat.py# Generic OpenAI client (Ollama/OpenRouter/Meta API)
│   │   ├── llm_mock.py         # Deterministic mock LLM
│   │   ├── seg_mock.py         # OpenCV color/contour segmentation mock
│   │   ├── seg_local_sam.py    # Stub with instructions for facebookresearch/sam3
│   │   └── seg_meta_api.py     # Stub pointing to https://dev.meta.ai/docs/overview
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── sop_builder.py      # SOP prompt extraction and schema validation
│   │   ├── grounding.py        # Object grounding and reference overlays
│   │   ├── checker.py          # Rule evaluation and step state machine
│   │   └── report.py           # Markdown and JSON report generator
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py    # 3-tab Streamlit dashboard
├── scripts/
│   └── make_synthetic_video.py # Synthetic video dataset generator
└── tests/                      # Pytest suite (22 tests passing)
```

---

## ⚖️ License
MIT License.
