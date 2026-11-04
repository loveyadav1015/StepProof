# Recording Checklist for StepProof Video Submission

Follow this checklist before hitting **Record** to ensure zero-glitch, high-reproducibility recording:

---

## 🖥️ Screen & Environment Setup
- [ ] **Display Resolution**: Set capture window or monitor to standard **1080p (1920x1080)** or 720p.
- [ ] **Browser Zoom**: Set browser zoom to **100%** (or 110% if projector/demo mode fonts need extra punch).
- [ ] **Clean Desktop**: Close unrelated tabs, messaging apps, and terminal windows.
- [ ] **Notifications Disabled**: Turn on OS "Do Not Disturb" / Focus Mode (disable desktop notifications, Slack, email alerts).
- [ ] **Audio & Microphone**:
  - Test microphone levels (ensure no fan noise, echo, or clipping).
  - Speak in a calm, confident, steady cadence matching [docs/DEMO_SCRIPT.md](file:///home/love/coding/meta-ai-hackathon/StepProof/docs/DEMO_SCRIPT.md).

---

## 🔒 Security & Privacy Checklist
- [ ] **No Secrets on Screen**: Ensure `.env` is never opened or shown on screen.
- [ ] **Terminal Cleared**: Run `clear` in terminal before switching to console views.
- [ ] **Working Directory Clean**: Run `make secrets-check` to confirm no live credentials reside in tracked files.

---

## ⚙️ Technical Dry Run
- [ ] **Rehearsal**: Run `make demo-record` once to pre-load all assets and cache entries.
- [ ] **Cache Only Mode**: Run `make quickstart` in an isolated shell to ensure the Streamlit app loads immediately without internet or API key prompts.
- [ ] **Tabs Sequence Verification**:
  1. Tab 0: Demo Mode (Large fonts, projector mode).
  2. Tab 1: Build SOP (Expert video & step cards).
  3. Tab 2: Check Run (Baseline correct & skipped step alert).
  4. Tab 3: Report & Audit (Provenance banner and downloadable reports).

---

## 📹 Post-Recording Verification
- [ ] Video duration is between **2:30 and 3:00 minutes** (under official cutoff limits).
- [ ] Voice audio is clear and synchronized with screen actions.
- [ ] Video resolution is sharp (at least 1080p landscape).
- [ ] Upload to YouTube (Unlisted), Vimeo, or Google Drive, and copy the sharing URL into `docs/SUBMISSION.md` and `docs/site/index.html`.
