# StepProof Demonstration Video Script (3-Minute Timed Walkthrough)

**Audience:** Hackathon judges, technical evaluators, manufacturing engineers  
**Total Target Time:** 3 minutes (180 seconds)  
**Tone:** Concise, credible, technical, calm.  

---

## ⏱️ Timeline & Segment Breakdown

### [0:00 - 0:20] 1. The Problem (20 seconds)
- **Visual:** Open browser showing the landing page or a physical workbench overview.
- **Narrator Voiceover:**
  > *"Industrial assembly and repair work still rely on printed PDFs or static checklists that nobody reads in real time. When a technician skips a resistor, wires a component out of sequence, or grabs the wrong part, the error is often only caught at end-of-line testing—or by a customer. StepProof solves this by turning a single demonstration video into an active verification engine."*

---

### [0:20 - 0:50] 2. Learn from One Video: Expert to SOP (30 seconds)
- **Action:** Switch to Streamlit dashboard (`make app` or `make demo-record`), click on **"Tab 1: Build SOP"**.
- **Visual:** Point to the expert demonstration player and click **"Generate Golden SOP"**.
- **Narrator Voiceover:**
  > *"We don't need hundreds of training examples. StepProof takes one clean demonstration video. A multimodal VLM extracts atomic, ordered assembly steps, identifying key objects, expected physical states, and common fault modes. Here you see our structured SOP schema generated with zero-shot spatial bounding from SAM 3.1."*
- **What to point at:** The structured step cards showing `Step 1: Place red block`, `Step 2: Place green block`, and the segmentation overlays with area metrics.

---

### [0:50 - 1:20] 3. Correct Learner Execution (30 seconds)
- **Action:** Switch to **"Tab 2: Check Run"**, select the baseline correct execution video, and click **"Verify Learner Video"**.
- **Visual:** Show the progression and the green checklist output.
- **Narrator Voiceover:**
  > *"When a learner follows the procedure correctly, StepProof uses multi-frame temporal smoothing so that brief hand occlusions or tool shadows don't trigger false alarms. All steps transition from PENDING to DONE with 100% confidence and zero false alarms."*
- **What to point at:** The green success alert and the step timeline showing all steps verified.

---

### [1:20 - 2:20] 4. Faulty Execution & Staged Errors (60 seconds)
- **Action:** Select `Learner 2: Skipped Step`, click **"Verify"**.
- **Visual:** An alert appears with red banners and an evidence frame overlay.
- **Narrator Voiceover:**
  > *"Now watch what happens when a learner makes an error. Here, the operator skipped Step 2 and jumped directly to Step 3. StepProof's deterministic state machine immediately identifies the sequence violation. It flags the step with a primary fault type: SKIPPED."*
- **Action:** Show the evidence frame with the red bounding box and missing component highlight.
- **Narrator Voiceover:**
  > *"Crucially, it captures and saves the exact video frame with overlay bounding boxes, providing an auditable visual evidence record for shop-floor supervisors. The same engine distinguishes out-of-order execution from wrong part substitution."*
- **What to point at:** The side-by-side Evidence Panel: Model Reason, Failed Rule code block, and the visual overlay frame showing the missing block.

---

### [2:20 - 2:50] 5. Meta Model Architecture & Measured Results (30 seconds)
- **Action:** Click on **"Tab 3: Report & Audit"** or show `eval_report.md`.
- **Visual:** Point to the Provenance Banner and the Model IDs.
- **Narrator Voiceover:**
  > *"Under the hood, StepProof relies on Meta's multimodal stack: Muse Wonder 1.0 for procedure decomposition and ambiguity judgment, and Segment Anything (SAM 3.1) for zero-shot object boundary extraction. Because weights can run on-premise, sensitive footage never has to leave the factory floor. Every run maintains cryptographic provenance, tracking token consumption and cost down to fractions of a cent."*
- **What to point at:** The provenance badges (`Meta model` vs `Dev stand-in`), the per-step latency stats, and cost tracking.

---

### [2:50 - 3:00] 6. Summary & Next Steps (10 seconds)
- **Visual:** Return to the GitHub repository / documentation page.
- **Narrator Voiceover:**
  > *"StepProof brings automated, verifiable quality assurance to every workshop. Judges can clone our repository and run `make quickstart` to replay this entire verification bundle offline in seconds. Thank you."*
