# StepProof Evaluation Report: Synthetic Block Assembly Verification Suite

**Task:** Colored Block Assembly
**SOP Step Count:** 4
**Manifest:** `manifests/synthetic_manifest.yaml`

## 1. Executive Summary & Quality Metrics

| Metric | Measured Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **Correct-Run False Alarms** | 0 | 0 | Optimal |
| **Detection Rate (Staged Errors)** | 100.0% | 100% | Optimal |
| **Average Latency Per Step** | 0.013s | < 0.500s (mock) | Good |
| **Estimated Cost Per Run** | $0.000000 USD | < $0.05 | Low Cost |

## 2. Per-Scenario Verification Outcomes

| Scenario Name | Result | Expected Flags | Observed Flags | Observed Types | Latency / Step | Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Correct Sequence Execution | PASS | None | None | `none` | 0.010s | $0.000000 |
| Step 2 Skipped (Missing Green Block) | PASS | Steps [2] | Steps [2] | `2:skipped` | 0.016s | $0.000000 |
| Step 2 Wrong Part (Purple Block) | PASS | Steps [2] | Steps [2] | `2:wrong` | 0.014s | $0.000000 |
| Step 2 Out-of-Order (Blue Placed First) | PASS | Steps [2] | Steps [2] | `2:out_of_order` | 0.011s | $0.000000 |

## 3. Analysis and Recommendations

- **Out of Order vs Skipped**: The state machine maintains active evaluation across all steps so that steps completed out of sequence are distinguished cleanly from steps never attempted.
- **Wrong Part Recognition**: Intruder objects and common error candidates are checked against reference object bounds and rule checks, preventing misclassification.
