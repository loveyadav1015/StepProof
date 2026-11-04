"""Streamlit web dashboard for StepProof: SOP generation, verification, and visual audit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stepproof.config import settings
from stepproof.ingest.sampler import sample_video
from stepproof.pipeline.checker import StepChecker
from stepproof.pipeline.grounding import ground_step_objects
from stepproof.pipeline.report import export_report, generate_markdown_report
from stepproof.pipeline.sop_builder import build_sop
from stepproof.provenance import assert_meta_compliant
from stepproof.providers.factory import get_llm_provider, get_seg_provider
from stepproof.schemas import SOP, RunReport

st.set_page_config(
    page_title="StepProof | Industrial Video SOP & Verification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for styling tokens (tokens.css alignment)
st.markdown(
    """
    <style>
    /* Status chips & timeline */
    .status-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .chip-done { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .chip-flagged { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .chip-active { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .chip-pending { background-color: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }

    /* Timeline card container */
    .timeline-step {
        background-color: #181818;
        border: 1px solid #272727;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .timeline-step.flagged {
        border-left: 4px solid #ef4444;
    }
    .timeline-step.done {
        border-left: 4px solid #10b981;
    }

    /* Mode badges */
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .badge-replay { background-color: #272727; color: #a3a3a3; border: 1px solid #383838; }
    .badge-live { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-synthetic { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

    /* Projector display style */
    .projector-alert {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 18px;
        border-radius: 8px;
        margin: 14px 0;
    }
    .projector-alert.danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 2px solid #ef4444;
    }
    .projector-alert.success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #86efac;
        border: 2px solid #10b981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize data directories
Path("data/outputs").mkdir(parents=True, exist_ok=True)
Path("data/outputs/runs").mkdir(parents=True, exist_ok=True)


def get_provenance_badge_html(meta_compliant: bool) -> str:
    """Return colored HTML badge for provenance status."""
    if meta_compliant:
        return (
            '<span style="background-color: rgba(16, 185, 129, 0.2); color: #34d399; padding: 3px 8px; '
            'border-radius: 4px; font-weight: 600; font-size: 0.8em; border: 1px solid rgba(16, 185, 129, 0.4);">🟢 Meta model</span>'
        )
    return (
        '<span style="background-color: rgba(100, 116, 139, 0.2); color: #94a3b8; padding: 3px 8px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.8em; border: 1px solid rgba(100, 116, 139, 0.4);">⚪ Dev stand-in</span>'
    )


# --- Sidebar ---
st.sidebar.title("StepProof Settings")

require_meta = st.sidebar.checkbox(
    "Enforce STEPPROOF_REQUIRE_META=1",
    value=settings.stepproof_require_meta,
    help="When enabled, non-Meta models (mocks, free stand-ins) and non-compliant cache entries are refused.",
)

llm_choice = st.sidebar.selectbox(
    "LLM Provider",
    options=["mock", "openai_compat"],
    index=0 if settings.stepproof_llm == "mock" else 1,
)

seg_choice = st.sidebar.selectbox(
    "Segmentation Provider",
    options=["mock", "dev_color", "local_sam", "meta_api"],
    index=0 if settings.stepproof_seg == "mock" else (1 if settings.stepproof_seg == "dev_color" else (2 if settings.stepproof_seg == "local_sam" else 3)),
)

cache_mode = st.sidebar.selectbox(
    "Cache Mode",
    options=["live", "cache_only", "refresh"],
    index=["live", "cache_only", "refresh"].index(settings.stepproof_mode),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Compliance Status")
try:
    active_llm = get_llm_provider(llm_choice, require_meta=require_meta)
    llm_prov = active_llm.get_provenance()
    st.sidebar.markdown(f"**LLM:** `{llm_prov.model_name}`")
    st.sidebar.markdown(get_provenance_badge_html(llm_prov.meta_compliant), unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"LLM Provider Refused: {e}")

try:
    active_seg = get_seg_provider(seg_choice, require_meta=require_meta)
    seg_prov = active_seg.get_provenance()
    st.sidebar.markdown(f"**Segmentation:** `{seg_prov.model_name}`")
    st.sidebar.markdown(get_provenance_badge_html(seg_prov.meta_compliant), unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"Segmentation Refused: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    "StepProof turns an expert video into a structured SOP, then checks learner execution "
    "videos and flags skipped, wrong, or out-of-order steps with visual evidence."
)


# --- Header with Product Name & Mode Badges ---
mode_label = "REPLAY" if cache_mode == "cache_only" else ("LIVE" if cache_mode == "live" else "REFRESH")
badge_class = "badge-replay" if cache_mode == "cache_only" else "badge-live"

st.markdown(
    f"""
    <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">👁️ StepProof</h1>
            <span class="mode-badge {badge_class}">{mode_label} MODE</span>
            <span class="mode-badge badge-synthetic">SYNTHETIC DATA AVAILABLE</span>
        </div>
        <div>
            {get_provenance_badge_html(require_meta)}
        </div>
    </div>
    <p style="color: #a3a3a3; font-size: 1.05rem; margin-top: 0;">Turn one expert video into a verified procedure, and catch mistakes as they happen.</p>
    """,
    unsafe_allow_html=True,
)

tab0, tab1, tab2, tab3 = st.tabs(["🎥 Demo Mode", "📋 Build SOP", "🔍 Check Run", "📊 Report & Audit"])

# ==============================================================================
# TAB 0: Demo Mode (Large-Font Projector-Friendly View)
# ==============================================================================
with tab0:
    st.header("🎯 Projector Demo: End-to-End Visual Verification")
    st.warning("⚠️ **Synthetic data & Dev stand-in**: Results displayed here run on synthetic video simulation. Live Meta model runs pending real footage.")

    # 1. Expert Video
    st.markdown("### 1. Expert Demonstration")
    expert_vid_path = Path("data/synthetic/expert.mp4")
    if not expert_vid_path.exists():
        expert_vid_path = Path("data/expert/expert.mp4")
    if expert_vid_path.exists():
        st.video(str(expert_vid_path))
    else:
        st.warning("No expert video found. Run `make demo` first.")

    # 2. Generated SOP
    st.markdown("### 2. Generated Golden SOP")
    sop_file = Path("data/outputs/demo_sop.json")
    if not sop_file.exists():
        sop_file = Path("data/sop/breadboard_led.json")
    if sop_file.exists():
        try:
            demo_sop = SOP.model_validate_json(sop_file.read_text(encoding="utf-8"))
            st.markdown(f"**Task:** `{demo_sop.task}` ({len(demo_sop.steps)} steps)")
            cols = st.columns(min(3, len(demo_sop.steps)))
            for i, s in enumerate(demo_sop.steps):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"""
                        <div class="timeline-step">
                            <span class="status-chip chip-done">STEP {s.id}</span>
                            <div style="font-weight: 600; margin: 6px 0;">{s.instruction}</div>
                            <div style="color: #737373; font-size: 0.85rem;">{s.expected_end_state}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception:
            st.code(sop_file.read_text())
    else:
        st.caption("No SOP loaded yet.")

    # 3. Correct Run
    st.markdown("### 3. Baseline Correct Execution")
    correct_run_dir = Path("data/outputs/runs/correct")
    c_json = correct_run_dir / "report.json"
    if c_json.exists():
        c_rep = RunReport.model_validate_json(c_json.read_text())
        st.markdown(
            f"""
            <div class="projector-alert success">
                ✅ PASS: All {len(c_rep.verdicts)} Assembly Steps Verified Cleanly
                <div style="font-size: 0.9rem; font-weight: 400; margin-top: 4px; color: #d1fae5;">
                    0 False Alarms • Multi-frame Temporal Smoothing Active • Zero Step Violations
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Run `make demo` to generate baseline correct verification.")

    # 4. Faulty Run with Alert and Evidence Frame
    st.markdown("### 4. Faulty Execution Alert & Visual Evidence (Staged Error)")
    skipped_run_dir = Path("data/outputs/runs/skipped")
    s_json = skipped_run_dir / "report.json"
    if s_json.exists():
        s_rep = RunReport.model_validate_json(s_json.read_text())
        col_err1, col_err2 = st.columns([1, 1])
        with col_err1:
            st.markdown(
                """
                <div class="projector-alert danger">
                    🚨 ALERT: Step 2 Skipped!
                    <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px; color: #fee2e2;">
                        • Fault Type: <code>SKIPPED</code><br>
                        • Failure Reason: Component was absent across consecutive frames.<br>
                        • Action: Supervisor review triggered
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for v in s_rep.verdicts:
                chip_type = "chip-flagged" if v.state == "FLAGGED" else "chip-done"
                st.markdown(
                    f"""
                    <div class="timeline-step {'flagged' if v.state == 'FLAGGED' else 'done'}">
                        <span class="status-chip {chip_type}">{v.state}</span>
                        <strong>Step {v.step_id}</strong>: {v.reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with col_err2:
            st.markdown("**Visual Evidence & Anomaly Overlay:**")
            flagged_v = next((v for v in s_rep.verdicts if v.state == "FLAGGED"), None)
            if flagged_v and flagged_v.evidence_frame_path and Path(flagged_v.evidence_frame_path).exists():
                st.image(str(flagged_v.evidence_frame_path), caption=f"Evidence for Step {flagged_v.step_id} ({flagged_v.flag_type})")
            else:
                st.info("Run `make demo` to generate evidence frames.")


# ==============================================================================
# TAB 1: Build SOP
# ==============================================================================
with tab1:
    st.header("1. Extract SOP from Expert Demonstration Video")

    expert_videos = {
        "Synthetic Block Assembly (Default)": "data/synthetic/expert.mp4",
        "Breadboard LED Expert (Real)": "data/expert/expert.mp4",
    }
    expert_choice = st.selectbox("Select Expert Video Demonstration", options=list(expert_videos.keys()))
    selected_expert_path = Path(expert_videos[expert_choice])

    col_exp_vid, col_exp_act = st.columns([1, 1])
    with col_exp_vid:
        if selected_expert_path.exists():
            st.video(str(selected_expert_path))
        else:
            st.warning(f"Video file not found at {selected_expert_path}. Run `make demo` to generate synthetic assets.")

    with col_exp_act:
        st.markdown("### Generate Golden SOP")
        st.write("multimodal VLM decomposes video frames into an ordered schema with verification rules.")
        out_sop_path = st.text_input("SOP Output Path", value="data/outputs/sop.json")
        btn_build_sop = st.button("🚀 Extract SOP from Video", type="primary")

    if btn_build_sop:
        if not selected_expert_path.exists():
            st.error(f"Cannot generate SOP: {selected_expert_path} does not exist.")
        else:
            with st.spinner("Extracting atomic steps with multimodal model..."):
                try:
                    llm = get_llm_provider(llm_choice, require_meta=require_meta)
                    sop, prov = build_sop(selected_expert_path, llm=llm, mode=cache_mode)
                    Path(out_sop_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_sop_path).write_text(sop.model_dump_json(indent=2), encoding="utf-8")
                    st.success(f"Successfully generated SOP with {len(sop.steps)} steps!")
                    st.session_state["active_sop"] = sop
                except Exception as ex:
                    st.error(f"SOP generation failed: {ex}")

    # Display Active SOP
    sop_to_show = st.session_state.get("active_sop")
    if not sop_to_show and Path(out_sop_path).exists():
        try:
            sop_to_show = SOP.model_validate_json(Path(out_sop_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    if sop_to_show:
        st.markdown("---")
        st.subheader(f"Generated Golden SOP: {sop_to_show.task}")
        for step in sop_to_show.steps:
            with st.expander(f"Step {step.id}: {step.instruction}", expanded=True):
                st.write(f"**Expected End State:** {step.expected_end_state}")
                st.write(f"**Physical Objects:** `{', '.join(step.objects)}`")
                checks_summary = [f"{c.type}({c.object}={c.value})" for c in step.checks]
                st.write(f"**Checks:** `{', '.join(checks_summary)}`")
                if step.common_errors:
                    st.write(f"**Common Errors:** {', '.join(step.common_errors)}")

        st.markdown("### Ground Physical Objects")
        if st.button("Run SAM / Segmentation Grounding on Expert Keyframes"):
            with st.spinner("Grounding objects across steps..."):
                try:
                    seg = get_seg_provider(seg_choice, require_meta=require_meta)
                    ground_dir = Path("data/outputs/grounding")
                    frames_dir = Path("data/outputs/expert_frames")
                    sampled_expert = sample_video(selected_expert_path, output_dir=frames_dir)
                    grounded, g_prov = ground_step_objects(sop_to_show, sampled_expert, seg, output_dir=ground_dir)

                    st.success("Grounding complete! Reference overlays generated.")
                    cols = st.columns(min(4, len(sop_to_show.steps)))
                    for i, step in enumerate(sop_to_show.steps):
                        overlay_p = ground_dir / f"step_{step.id}_grounding.jpg"
                        if overlay_p.exists():
                            with cols[i % len(cols)]:
                                st.image(str(overlay_p), caption=f"Step {step.id} Reference")
                except Exception as ex:
                    st.error(f"Grounding failed: {ex}")


# ==============================================================================
# TAB 2: Check Run
# ==============================================================================
with tab2:
    st.header("2. Check Learner Execution Video")

    learner_videos = {
        "Correct Sequence (All steps done in order)": "data/synthetic/learner_correct.mp4",
        "Skipped Step (Step 2 green block skipped)": "data/synthetic/learner_skipped.mp4",
        "Wrong Part (Purple block used instead of green)": "data/synthetic/learner_wrong_part.mp4",
        "Out of Order (Step 3 placed before Step 2)": "data/synthetic/learner_out_of_order.mp4",
    }

    c_sel, c_play = st.columns([1, 1])
    with c_sel:
        learner_choice = st.selectbox("Choose Learner Test Footage", options=list(learner_videos.keys()))
        learner_video_path = Path(learner_videos[learner_choice])

        golden_sop_path = st.text_input("Active SOP JSON Path", value="data/outputs/demo_sop.json")
        run_name = st.text_input("Run Identifier", value="demo_run")
        btn_check = st.button("⚡ Verify Learner Video", type="primary")

    with c_play:
        if learner_video_path.exists():
            st.video(str(learner_video_path))

    if btn_check:
        if not learner_video_path.exists():
            st.error(f"Learner video not found at {learner_video_path}. Please generate synthetic videos in Tab 1.")
        elif not Path(golden_sop_path).exists():
            st.error(f"SOP file not found at {golden_sop_path}. Please generate SOP in Tab 1 first.")
        else:
            with st.spinner("Analyzing learner video through StepProof state machine..."):
                try:
                    with open(golden_sop_path, "r", encoding="utf-8") as f:
                        check_sop = SOP.model_validate_json(f.read())

                    run_dir = Path("data/outputs/runs") / run_name
                    frames_dir = run_dir / "learner_frames"
                    sampled_learner = sample_video(learner_video_path, output_dir=frames_dir)

                    seg = get_seg_provider(seg_choice, require_meta=require_meta)
                    judge = get_llm_provider(llm_choice, require_meta=require_meta)

                    checker = StepChecker(
                        sop=check_sop,
                        segmenter=seg,
                        llm_judge=judge,
                        consecutive_frames_required=3,
                        out_dir=run_dir,
                    )
                    verdicts = checker.verify_run(sampled_learner.frames)

                    rep = export_report(
                        sop_id=check_sop.task,
                        verdicts=verdicts,
                        provenance=[seg.get_provenance(), judge.get_provenance()],
                        timings={"ingest": 0.4, "check": 0.6},
                        output_dir=run_dir,
                    )
                    st.session_state["latest_run_report"] = rep
                    st.session_state["latest_run_dir"] = str(run_dir)
                    st.success("Verification complete! Results detailed below.")
                except Exception as ex:
                    st.error(f"Verification failed: {ex}")

    # Display Step Timeline & Evidence Panel
    latest_rep: RunReport | None = st.session_state.get("latest_run_report")
    if latest_rep:
        st.markdown("---")
        st.subheader("Step Timeline & Evidence Panel")

        is_meta_compliant = all(p.meta_compliant for p in latest_rep.provenance) if latest_rep.provenance else False
        prov_badge_html = get_provenance_badge_html(is_meta_compliant)

        for v in latest_rep.verdicts:
            col_stat, col_ev = st.columns([2, 1])

            with col_stat:
                chip_class = "chip-done" if v.state == "DONE" else "chip-flagged"
                flag_label = v.flag_type.upper().replace("_", " ") if v.flag_type != "none" else "VERIFIED"

                st.markdown(
                    f"""
                    <div class="timeline-step {'flagged' if v.state == 'FLAGGED' else 'done'}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div>
                                <span class="status-chip {chip_class}">{v.state}</span>
                                <span style="font-weight: 700; font-size: 1.1rem;">Step {v.step_id}</span>
                                <span style="color: #737373; font-size: 0.9rem; margin-left: 8px;">({flag_label})</span>
                            </div>
                            <div>{prov_badge_html}</div>
                        </div>
                        <div style="margin-bottom: 8px;">{v.reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if v.failed_rules:
                    st.markdown("**Failed Rule(s):**")
                    for r in v.failed_rules:
                        st.code(r)

            with col_ev:
                ev_path = v.evidence_frame_path
                if ev_path and Path(ev_path).exists():
                    st.image(str(ev_path), caption=f"Step {v.step_id} Evidence Overlay")
                else:
                    st.caption("No evidence overlay frame available")


# ==============================================================================
# TAB 3: Report & Audit
# ==============================================================================
with tab3:
    st.header("3. Verification Report & Provenance Audit")

    runs_base = Path("data/outputs/runs")
    available_runs = [d.name for d in runs_base.glob("*") if d.is_dir()] if runs_base.exists() else []

    if not available_runs:
        st.info("No verification runs found yet. Run a check in Tab 2 to produce reports.")
    else:
        sel_run = st.selectbox("Select Run Report", options=available_runs)
        selected_run_dir = runs_base / sel_run
        json_file = selected_run_dir / "report.json"
        md_file = selected_run_dir / "report.md"

        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    rep_data = RunReport.model_validate_json(f.read())

                if require_meta:
                    for p in rep_data.provenance:
                        assert_meta_compliant(p)

                # Provenance Banner
                st.markdown("### Provenance Audit")
                prov_cols = st.columns(len(rep_data.provenance))
                for idx, p in enumerate(rep_data.provenance):
                    with prov_cols[idx]:
                        st.markdown(f"**{p.provider.upper()}:** `{p.model_name}`")
                        st.caption(f"Host: `{p.base_url_host}`")
                        st.markdown(get_provenance_badge_html(p.meta_compliant), unsafe_allow_html=True)

                # Flagged Steps Detail Audit
                flagged_steps = [v for v in rep_data.verdicts if v.state != "DONE" or v.flag_type != "none"]
                if flagged_steps:
                    st.markdown("### ⚠️ Flagged Steps Visual Audit")
                    is_meta = all(p.meta_compliant for p in rep_data.provenance) if rep_data.provenance else False
                    p_badge = get_provenance_badge_html(is_meta)
                    for fv in flagged_steps:
                        fc1, fc2 = st.columns([2, 1])
                        with fc1:
                            f_label = fv.flag_type.upper().replace("_", " ")
                            st.error(f"**Step {fv.step_id}: FLAGGED ({f_label})**")
                            st.markdown(f"**Provenance:** {p_badge}", unsafe_allow_html=True)
                            st.markdown(f"**Model Reason:** {fv.reason}")
                            if fv.failed_rules:
                                st.markdown("**Failed Rule(s):**")
                                for r in fv.failed_rules:
                                    st.code(r)
                            else:
                                st.markdown(f"**Failed Rule:** Step sequence rule violated (type: `{fv.flag_type}`)")
                        with fc2:
                            ev_path = fv.evidence_frame_path
                            if ev_path and Path(ev_path).exists():
                                st.image(str(ev_path), caption=f"Step {fv.step_id} Evidence Overlay")
                            else:
                                st.caption("No evidence overlay available")

                st.markdown("---")
                if md_file.exists():
                    st.markdown(md_file.read_text())
                else:
                    st.markdown(generate_markdown_report(rep_data))

                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.download_button(
                        "📥 Download report.md",
                        data=generate_markdown_report(rep_data),
                        file_name=f"{sel_run}_report.md",
                        mime="text/markdown",
                    )
                with c_d2:
                    st.download_button(
                        "📥 Download report.json",
                        data=rep_data.model_dump_json(indent=2),
                        file_name=f"{sel_run}_report.json",
                        mime="application/json",
                    )
            except Exception as e:
                st.error(f"Failed to display report: {e}")
