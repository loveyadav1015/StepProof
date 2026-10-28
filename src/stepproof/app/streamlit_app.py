"""Streamlit web dashboard for StepProof: SOP generation, verification, and visual audit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stepproof.cache import SQLiteCache
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
    page_title="StepProof | Video SOP & Execution Verification",
    page_icon="👁️",
    layout="wide",
)

# Initialize data directories
Path("data/outputs").mkdir(parents=True, exist_ok=True)
Path("data/outputs/runs").mkdir(parents=True, exist_ok=True)


def get_provenance_badge_html(meta_compliant: bool) -> str:
    """Return colored HTML badge for provenance status."""
    if meta_compliant:
        return (
            '<span style="background-color: #2e7d32; color: white; padding: 3px 8px; '
            'border-radius: 4px; font-weight: bold; font-size: 0.85em;">🟢 Meta model</span>'
        )
    return (
        '<span style="background-color: #757575; color: white; padding: 3px 8px; '
        'border-radius: 4px; font-weight: bold; font-size: 0.85em;">⚪ Dev stand-in</span>'
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
    options=["mock", "local_sam", "meta_api"],
    index=0 if settings.stepproof_seg == "mock" else (1 if settings.stepproof_seg == "local_sam" else 2),
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


# --- Main Dashboard ---
st.title("👁️ StepProof: Video Verification Engine")
st.caption("Offline-first, multimodal video SOP generation and execution verification.")

tab1, tab2, tab3 = st.tabs(["📋 Build SOP", "🔍 Check Run", "📊 Report & Audit"])


# ==============================================================================
# TAB 1: Build SOP
# ==============================================================================
with tab1:
    st.header("1. Extract Golden SOP from Expert Video")

    col_v, col_opt = st.columns([2, 1])

    with col_v:
        synthetic_expert = Path("data/expert/expert.mp4")
        expert_options = ["Synthetic Expert: Block Assembly (data/expert/expert.mp4)"]
        expert_choice = st.selectbox("Select Expert Video", options=expert_options)
        selected_expert_path = synthetic_expert

        if not selected_expert_path.exists():
            st.warning("Synthetic videos not found. Click below to generate them.")
            if st.button("Generate Synthetic Demonstration Videos"):
                from scripts.make_synthetic_video import generate_all_synthetic_videos
                generate_all_synthetic_videos()
                st.success("Generated synthetic demo videos in data/expert and data/learner!")
                st.rerun()

        if selected_expert_path.exists():
            st.video(str(selected_expert_path))

    with col_opt:
        st.subheader("SOP Extraction Controls")
        out_sop_file = st.text_input("SOP Output Path", value="data/outputs/sop.json")
        btn_build = st.button("🚀 Generate Golden SOP", type="primary")

    if btn_build and selected_expert_path.exists():
        with st.spinner("Extracting frames (~2 fps) and building SOP..."):
            try:
                frames_dir = Path("data/outputs/expert_frames")
                sampled = sample_video(selected_expert_path, output_dir=frames_dir)
                cache = SQLiteCache()
                llm = get_llm_provider(llm_choice, require_meta=require_meta)

                sop, prov = build_sop(
                    sampled_video=sampled,
                    llm_provider=llm,
                    output_path=out_sop_file,
                    cache=cache,
                    mode=cache_mode,
                )
                st.session_state["active_sop"] = sop
                st.success(f"Generated SOP with {len(sop.steps)} steps!")
            except Exception as ex:
                st.error(f"SOP Generation failed: {ex}")

    # Display active SOP
    sop_to_show = st.session_state.get("active_sop")
    if not sop_to_show and Path(out_sop_file).exists():
        try:
            with open(out_sop_file, "r", encoding="utf-8") as f:
                sop_to_show = SOP.model_validate_json(f.read())
        except Exception:
            pass

    if sop_to_show:
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(f"Golden SOP: {sop_to_show.task}")
        with c2:
            st.markdown(get_provenance_badge_html(sop_to_show.created_with.meta_compliant), unsafe_allow_html=True)

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
        "Correct Sequence (All steps done in order)": "data/learner/learner_correct.mp4",
        "Skipped Step (Step 2 green block skipped)": "data/learner/learner_skipped.mp4",
        "Wrong Part (Purple block used instead of green)": "data/learner/learner_wrong_part.mp4",
        "Out of Order (Step 3 placed before Step 2)": "data/learner/learner_out_of_order.mp4",
    }

    c_sel, c_play = st.columns([1, 1])
    with c_sel:
        learner_choice = st.selectbox("Choose Learner Test Footage", options=list(learner_videos.keys()))
        learner_video_path = Path(learner_videos[learner_choice])

        golden_sop_path = st.text_input("Active SOP JSON Path", value="data/outputs/sop.json")
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

    # Display Verdicts & Checklist
    latest_rep: RunReport | None = st.session_state.get("latest_run_report")
    if latest_rep:
        st.markdown("---")
        st.subheader("Step Verification Checklist")

        is_meta_compliant = all(p.meta_compliant for p in latest_rep.provenance) if latest_rep.provenance else False
        prov_badge_html = get_provenance_badge_html(is_meta_compliant)

        for v in latest_rep.verdicts:
            col_stat, col_ev = st.columns([2, 1])

            with col_stat:
                if v.state == "DONE":
                    st.success(f"**Step {v.step_id}: DONE** — {v.reason}")
                else:
                    flag_label = v.flag_type.upper().replace("_", " ")
                    st.error(f"**Step {v.step_id}: FLAGGED ({flag_label})**")
                    st.markdown(f"**Provenance:** {prov_badge_html}", unsafe_allow_html=True)
                    st.markdown(f"**Model Reason:** {v.reason}")

                    if v.failed_rules:
                        st.markdown("**Failed Rule(s):**")
                        for r in v.failed_rules:
                            st.code(r)
                    else:
                        st.markdown(f"**Failed Rule:** Step sequence rule violated (type: `{v.flag_type}`)")

            with col_ev:
                ev_path = v.evidence_frame_path
                if ev_path and Path(ev_path).exists():
                    st.image(str(ev_path), caption=f"Step {v.step_id} Evidence Overlay")
                else:
                    st.caption("No evidence frame available")


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
