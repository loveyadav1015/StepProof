VENV ?= .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff
STREAMLIT = $(VENV)/bin/streamlit

.PHONY: all install test lint app demo demo-cached final clean help

help:
	@echo "StepProof - Industrial Video SOP & Execution Verification"
	@echo "Targets:"
	@echo "  make install     - Create venv and install package in editable mode with dev dependencies"
	@echo "  make test        - Run pytest suite"
	@echo "  make lint        - Run ruff linter check"
	@echo "  make app         - Run Streamlit UI"
	@echo "  make demo        - Generate synthetic videos and run full verification pipeline"
	@echo "  make demo-cached - Replay previous run strictly from cache (STEPPROOF_MODE=cache_only)"
	@echo "  make final       - Run with STEPPROOF_REQUIRE_META=1 (refuses non-Meta models)"
	@echo "  make clean       - Remove cached files and test artifacts"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtualenv with python3.11..."; \
		python3.11 -m venv $(VENV); \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@mkdir -p data/expert data/learner data/cache data/outputs

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check .

app:
	$(STREAMLIT) run src/stepproof/app/streamlit_app.py

demo:
	$(PYTHON) scripts/make_synthetic_video.py
	@echo "Running StepProof mock verification demo..."
	$(PYTHON) -m stepproof build-sop data/expert/expert.mp4 --out data/outputs/demo_sop.json
	$(PYTHON) -m stepproof ground data/expert/expert.mp4 --sop data/outputs/demo_sop.json --out-dir data/outputs/grounding
	@echo "\n--- Checking Correct Learner Video ---"
	$(PYTHON) -m stepproof check data/learner/learner_correct.mp4 --sop data/outputs/demo_sop.json --out-dir data/outputs/runs/correct
	@echo "\n--- Checking Skipped-Step Learner Video ---"
	$(PYTHON) -m stepproof check data/learner/learner_skipped.mp4 --sop data/outputs/demo_sop.json --out-dir data/outputs/runs/skipped
	@echo "\n--- Checking Wrong-Part Learner Video ---"
	$(PYTHON) -m stepproof check data/learner/learner_wrong_part.mp4 --sop data/outputs/demo_sop.json --out-dir data/outputs/runs/wrong_part
	@echo "\n--- Checking Out-of-Order Learner Video ---"
	$(PYTHON) -m stepproof check data/learner/learner_out_of_order.mp4 --sop data/outputs/demo_sop.json --out-dir data/outputs/runs/out_of_order
	@echo "\n--- Generating Final Markdown Report ---"
	$(PYTHON) -m stepproof report data/outputs/runs/skipped --out data/outputs/runs/skipped/report.md
	@echo "\nDemo finished successfully! Sample report generated at data/outputs/runs/skipped/report.md"

demo-cached:
	@echo "Replaying previous runs strictly from cache (STEPPROOF_MODE=cache_only)..."
	STEPPROOF_MODE=cache_only $(PYTHON) -m stepproof replay data/outputs/runs/correct
	STEPPROOF_MODE=cache_only $(PYTHON) -m stepproof replay data/outputs/runs/skipped
	STEPPROOF_MODE=cache_only $(PYTHON) -m stepproof replay data/outputs/runs/wrong_part
	STEPPROOF_MODE=cache_only $(PYTHON) -m stepproof replay data/outputs/runs/out_of_order
	@echo "\nCached replay completed successfully!"

final:
	@echo "Checking Meta compliance enforcement (STEPPROOF_REQUIRE_META=1)..."
	@STEPPROOF_REQUIRE_META=1 $(PYTHON) -m stepproof replay data/outputs/runs/skipped --require-meta || { \
		echo "\n================================================================================"; \
		echo "❌ STEPPROOF FINAL COMPLIANCE REFUSAL:"; \
		echo "STEPPROOF_REQUIRE_META=1 is set. Cached results were generated with dev stand-in"; \
		echo "(mock) models. Non-Meta model outputs are strictly refused for compliance."; \
		echo "To pass 'make final', configure real Meta credentials (api.meta.ai or local SAM)."; \
		echo "================================================================================\n"; \
		exit 1; \
	}


clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
