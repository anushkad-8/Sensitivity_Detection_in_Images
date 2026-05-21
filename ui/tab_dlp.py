"""
ui/tab_dlp.py
─────────────
OCR / NLP / Vision DLP tab — full dashboard layout.

Wires directly into main.run_pipeline().
Handles upload, options, analysis, and rich result display.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st
from PIL import Image

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.components import (
    render_risk_banner,
    render_findings_table,
    render_score_breakdown,
    render_pipeline_stages,
    section_label,
    render_empty_state,
    RISK_COLORS,
    RISK_ICONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_dlp():
    st.sidebar.markdown("""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: #4B5563;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    ">OCR · DLP OPTIONS</div>
    """, unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "Annotation Mode",
        options=["highlight", "redact"],
        index=0,
        help="highlight = colour box around findings | redact = black box covering text",
    )

    psm = st.sidebar.selectbox(
        "Tesseract PSM",
        options=[6, 3, 11],
        index=0,
        format_func=lambda x: {6: "6 — Block text (default)", 3: "3 — Auto layout", 11: "11 — Sparse / ID cards"}[x],
        help="Page segmentation mode. Use 11 for ID cards and forms.",
    )

    conf = st.sidebar.slider(
        "OCR Confidence Threshold",
        min_value=20, max_value=90, value=60, step=5,
        help="Words below this confidence are dropped. Lower = more words kept (noisier).",
    )

    st.sidebar.markdown("---")

    run_nlp = st.sidebar.toggle("NLP Classification", value=True,
        help="Enable context verification, document labels, and NER")
    run_ner = st.sidebar.toggle("NER Entity Detection", value=True,
        help="spaCy NER for person names, orgs, dates (requires NLP on)")
    run_vision = st.sidebar.toggle("Vision Classification", value=True,
        help="OpenCV/CLIP document type classifier + OCR failure detection")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.65rem;color:#374151;line-height:1.7;font-family:'JetBrains Mono',monospace;">
    <b style="color:#4B5563;">EVIDENCE FUSION POLICY</b><br>
    OCR/NLP = primary evidence<br>
    Vision = supporting signal<br>
    Heuristic-only = REVIEW flag<br>
    False positives → analyst queue
    </div>
    """, unsafe_allow_html=True)

    return mode, psm, conf, run_nlp, run_ner, run_vision


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TAB RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab_dlp():
    mode, psm, conf, run_nlp, run_ner, run_vision = render_sidebar_dlp()

    # ── Upload zone ────────────────────────────────────────────────────────────
    col_up, col_btn = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drop image here or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
            label_visibility="collapsed",
            key="dlp_uploader",
        )
    with col_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("⟶  RUN SCAN", use_container_width=True, key="dlp_scan")

    if not uploaded:
        render_empty_state("DLP")
        return

    # ── Run pipeline ───────────────────────────────────────────────────────────
    result = st.session_state.get("dlp_result")
    last_file = st.session_state.get("dlp_last_file")

    if scan_clicked or (result is None) or (last_file != uploaded.name):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded.name).suffix,
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            from main import run_pipeline

            with st.spinner(""):
                progress_bar = st.progress(0, text="Initialising pipeline…")

                # Patch stdout to update progress (lightweight approach)
                import io, contextlib
                buf = io.StringIO()
                stages_done = [0]

                def _step_hook(n):
                    stages_done[0] = n
                    progress_bar.progress(
                        min(n / 8, 1.0),
                        text=f"Step {n}/8 — processing…"
                    )

                # Monkey-patch print to detect step completions
                _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else print
                import builtins
                _orig_builtins_print = builtins.print

                def _tracking_print(*args, **kwargs):
                    msg = " ".join(str(a) for a in args)
                    for i in range(1, 9):
                        if f"[{i}] ✅" in msg or f"[{i}/8]" in msg:
                            _step_hook(i)
                            break
                    _orig_builtins_print(*args, **kwargs)

                builtins.print = _tracking_print
                try:
                    result = run_pipeline(
                        image_path = tmp_path,
                        mode       = mode,
                        psm        = psm,
                        conf       = conf,
                        run_nlp    = run_nlp,
                        run_ner    = run_ner and run_nlp,
                        run_vision = run_vision,
                        encrypt    = False,
                        debug      = False,
                    )
                finally:
                    builtins.print = _orig_builtins_print

                progress_bar.progress(1.0, text="Scan complete.")

            # Store annotated image bytes in result for display
            ann_path = result.get("annotated_path")
            if ann_path and os.path.exists(ann_path):
                with open(ann_path, "rb") as f:
                    result["_annotated_bytes"] = f.read()

            result["_original_path"] = tmp_path
            st.session_state["dlp_result"]    = result
            st.session_state["dlp_last_file"] = uploaded.name
            progress_bar.empty()

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            return

    if result is None:
        return

    # ── Results dashboard ──────────────────────────────────────────────────────
    scored = result.get("scored_result", {})
    overall_risk  = scored.get("overall_risk", "NONE")
    is_sensitive  = scored.get("is_sensitive", False)
    matches       = scored.get("matches", [])

    # Risk banner — full width
    render_risk_banner(overall_risk, len(matches), is_sensitive)

    # ── Main layout: images (left) | analysis (right) ──────────────────────────
    col_img, col_analysis = st.columns([1, 1], gap="large")

    with col_img:
        section_label("IMAGE PAIR")

        orig_path = result.get("_original_path")
        ann_bytes = result.get("_annotated_bytes")

        img_tab_orig, img_tab_ann = st.tabs(["Original", "Annotated"])

        with img_tab_orig:
            if orig_path and os.path.exists(orig_path):
                st.image(orig_path, use_container_width=True)
            else:
                st.info("Original image not available")

        with img_tab_ann:
            if ann_bytes:
                st.image(ann_bytes, use_container_width=True,
                         caption=f"Mode: {mode.upper()}")
            else:
                st.info("Annotated image not generated")

        # Vision result chip
        vision = result.get("vision_result", {})
        doc_type  = vision.get("document_type", "unknown")
        v_conf    = vision.get("type_confidence", 0.0)
        v_sens    = vision.get("sensitivity_level", "LOW")
        v_model   = vision.get("model_used", "—")
        v_ocr_fail= vision.get("ocr_failure_risk", False)

        vfg, _    = RISK_COLORS.get(v_sens, ("#8B9AC7", "#151B2D"))

        st.markdown(f"""
        <div style="
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            align-items: center;
        ">
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px;">VISION · DOC TYPE</div>
                <div style="font-size:0.85rem;font-weight:700;color:{vfg};letter-spacing:0.05em;">{doc_type.upper()}</div>
            </div>
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px;">CONFIDENCE</div>
                <div style="font-size:0.85rem;font-weight:600;color:#CBD5E1;">{v_conf:.2f}</div>
            </div>
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px;">OCR FAIL RISK</div>
                <div style="font-size:0.85rem;font-weight:600;color:{'#FF8C42' if v_ocr_fail else '#06D6A0'};">{'YES' if v_ocr_fail else 'NO'}</div>
            </div>
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px;">BACKEND</div>
                <div style="font-size:0.75rem;color:#8B9AC7;">{v_model}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_analysis:
        section_label("FINDINGS")
        render_findings_table(matches)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        render_score_breakdown(scored)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        render_pipeline_stages(result)

    # ── OCR text expander ──────────────────────────────────────────────────────
    ocr_text = result.get("ocr_text", "")
    with st.expander("RAW OCR TEXT", expanded=False):
        if ocr_text.strip():
            st.code(ocr_text, language=None)
        else:
            st.markdown(
                "<span style='color:#4B5563;font-size:0.8rem;'>OCR returned no text for this image.</span>",
                unsafe_allow_html=True
            )

    # ── Duration ──────────────────────────────────────────────────────────────
    duration = result.get("duration_sec", 0)
    st.markdown(f"""
    <div style="
        text-align: right;
        font-size: 0.65rem;
        color: #374151;
        margin-top: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    ">Scan completed in {duration}s · {uploaded.name}</div>
    """, unsafe_allow_html=True)