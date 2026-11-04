# StepProof Verification Report: Colored Block Assembly

**Run Status:** ⚠️ ISSUES DETECTED (3/4 steps passed, 1 flagged)

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| Total Steps | 4 |
| Steps Verified (DONE) | 3 |
| Steps Flagged | 1 |
| Total Pipeline Time | 0.12s |
| Stage Timing: ingest | 0.05s |
| Stage Timing: check | 0.07s |
| Estimated Run Cost | $0.0000 USD |

## 2. Model Provenance & Compliance

| Role / Provider | Model Name | Host Endpoint | Meta Badge | Timestamp (UTC) |
| :--- | :--- | :--- | :--- | :--- |
| `mock` | `opencv-color-contour-mock` | `local` | ⚪ *Dev stand-in* | 2026-10-28T21:04:47 |
| `mock` | `mock-vision-sop` | `localhost` | ⚪ *Dev stand-in* | 2026-10-28T21:04:47 |

## 3. Step Verdicts & Visual Evidence

### Step 1: ✅ **DONE**
- **Confidence:** 0.96
- **Reason:** Step 1 verified: all objects (red block) located in expected state.
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/wrong_part/step_1_done.jpg)`

### Step 2: ❌ **FLAGGED (WRONG)**
- **Confidence:** 0.95
- **Reason:** Failed: Missing required object 'green block', Count mismatch for 'green block': expected 1, found 0, Detected wrong part 'purple block' instead of 'green block'
- **Failed Rules:**
  - `Missing required object 'green block'`
  - `Count mismatch for 'green block': expected 1, found 0`
  - `Detected wrong part 'purple block' instead of 'green block'`
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/wrong_part/step_2_wrong.jpg)`

### Step 3: ✅ **DONE**
- **Confidence:** 0.96
- **Reason:** Step 3 verified: all objects (blue block) located in expected state.
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/wrong_part/step_3_done.jpg)`

### Step 4: ✅ **DONE**
- **Confidence:** 0.96
- **Reason:** Step 4 verified: all objects (yellow block) located in expected state.
- **Evidence Keyframe:** `![Evidence](data/outputs/runs/wrong_part/step_4_done.jpg)`
