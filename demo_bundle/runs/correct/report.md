# StepProof Verification Report: Colored Block Assembly

**Run Status:** ⚠️ ISSUES DETECTED (1/4 steps passed, 3 flagged)

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| Total Steps | 4 |
| Steps Verified (DONE) | 1 |
| Steps Flagged | 3 |
| Total Pipeline Time | 0.11s |
| Stage Timing: ingest | 0.06s |
| Stage Timing: check | 0.05s |
| Estimated Run Cost | $0.0000 USD |

## 2. Model Provenance & Compliance

| Role / Provider | Model Name | Host Endpoint | Meta Badge | Timestamp (UTC) |
| :--- | :--- | :--- | :--- | :--- |
| `mock` | `opencv-color-contour-mock` | `local` | ⚪ *Dev stand-in* | 2026-11-04T17:31:58 |
| `mock` | `mock-vision-sop` | `localhost` | ⚪ *Dev stand-in* | 2026-11-04T17:31:58 |

## 3. Step Verdicts & Visual Evidence

### Step 1: ✅ **DONE**
- **Confidence:** 0.96
- **Reason:** Step 1 verified: all objects (red block) located in expected state.
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/correct/step_1_done.jpg)`

### Step 2: ❌ **FLAGGED (WRONG)**
- **Confidence:** 0.95
- **Reason:** Failed at end of run: End-state not achieved before run ended
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/correct/step_2_wrong.jpg)`

### Step 3: ❌ **FLAGGED (WRONG)**
- **Confidence:** 0.95
- **Reason:** Failed at end of run: End-state not achieved before run ended
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/correct/step_3_wrong.jpg)`

### Step 4: ❌ **FLAGGED (WRONG)**
- **Confidence:** 0.95
- **Reason:** Failed at end of run: Missing required object 'yellow block' (min_area=500px), Count mismatch for 'yellow block': expected 1, found 0
- **Failed Rules:**
  - `Missing required object 'yellow block' (min_area=500px)`
  - `Count mismatch for 'yellow block': expected 1, found 0`
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/correct/step_4_wrong.jpg)`
