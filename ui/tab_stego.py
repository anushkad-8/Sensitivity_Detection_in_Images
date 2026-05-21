"""
ui/tab_stego.py
───────────────
Steganography Detection tab.

Architecture:
    This tab is designed as a clean wrapper around your steganography module.
    Wire it by replacing the _run_stego_analysis() stub with your actual
    module call — the entire UI, results layout, and reporting remain the same.

Extension points (search for "── WIRE YOUR MODULE HERE ──"):
    1. _run_stego_analysis()  — replace stub with your module's entry point
    2. STEGO_TECHNIQUES       — add/remove technique names as your module grows

The UI handles: upload, technique selection, analysis, findings display,
confidence breakdown, and a visual channel inspector for RGB/LSB planes.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import numpy as np

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.components import (
    section_label,
    render_empty_state,
    RISK_COLORS,
    RISK_ICONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEGO TECHNIQUES REGISTRY
# Add/remove entries here as your module grows.
# ─────────────────────────────────────────────────────────────────────────────

STEGO_TECHNIQUES = {
    "LSB Steganography"       : "lsb",
    "DCT Coefficient Analysis": "dct",
    "Metadata / EXIF Embed"   : "exif",
    "Palette Manipulation"    : "palette",
    "Chi-Square Attack"       : "chi_square",
    "RS Analysis"             : "rs",
}


# ─────────────────────────────────────────────────────────────────────────────
# ── WIRE YOUR MODULE HERE ── (1 of 2)
# Replace this function body with your steganography module's entry point.
# Expected return schema is documented below.
# ─────────────────────────────────────────────────────────────────────────────

def _run_stego_analysis(image_path: str, techniques: list) -> dict:
    """
    Stub — replace with your steganography module call.

    Example wiring:
        from modules.stego_detector import analyse
        return analyse(image_path, techniques=techniques)

    Expected return schema:
    {
        "is_stego"         : bool,
        "overall_risk"     : str,   # CRITICAL / HIGH / MEDIUM / LOW / REVIEW / NONE
        "overall_confidence: float, # 0.0 – 1.0
        "findings": [
            {
                "technique"  : str,   # e.g. "lsb"
                "label"      : str,   # e.g. "LSB Steganography"
                "detected"   : bool,
                "confidence" : float, # 0.0 – 1.0
                "risk_level" : str,
                "detail"     : str,   # human-readable finding summary
                "payload_hint": str | None,  # e.g. "~2.1KB estimated hidden data"
            },
            ...
        ],
        "image_stats": {
            "width"       : int,
            "height"      : int,
            "mode"        : str,   # RGB / RGBA / L etc.
            "file_size_kb": float,
            "entropy"     : float, # image entropy — high entropy is suspicious
        },
        "duration_sec" : float,
    }
    """
    import time
    import random

    # ── STUB IMPLEMENTATION (produces realistic-looking demo data) ────────────
    # Remove everything below and replace with your module call.
    time.sleep(0.6)   # simulate analysis time

    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        w, h = img.size
        mode = img.mode
        file_kb = os.path.getsize(image_path) / 1024

        # Entropy estimate
        import numpy as np_inner
        arr = np_inner.array(img.convert("L"))
        hist = np_inner.bincount(arr.ravel(), minlength=256)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy = float(-np_inner.sum(prob * np_inner.log2(prob)))
    except Exception:
        w, h, mode, file_kb, entropy = 0, 0, "?", 0.0, 0.0

    # Stub findings — deterministic based on filename for demo consistency
    seed = sum(ord(c) for c in image_path)
    rng  = random.Random(seed)

    findings = []
    for tech_key in techniques:
        label = next((k for k, v in STEGO_TECHNIQUES.items() if v == tech_key), tech_key)
        detected = rng.random() < 0.25   # 25% hit rate in stub
        conf = rng.uniform(0.6, 0.9) if detected else rng.uniform(0.05, 0.25)
        risk = "LOW" if not detected else ("HIGH" if conf > 0.75 else "MEDIUM")
        findings.append({
            "technique"   : tech_key,
            "label"       : label,
            "detected"    : detected,
            "confidence"  : round(conf, 3),
            "risk_level"  : risk,
            "detail"      : (
                f"Anomalous {tech_key.upper()} patterns detected with confidence {conf:.0%}. "
                "Possible hidden payload — manual inspection recommended."
            ) if detected else f"No {tech_key.upper()} anomalies detected.",
            "payload_hint": f"~{rng.uniform(0.5, 8.0):.1f} KB estimated" if detected else None,
        })

    any_detected = any(f["detected"] for f in findings)
    overall_conf = max((f["confidence"] for f in findings), default=0.0)
    overall_risk = "HIGH" if overall_conf > 0.75 and any_detected else (
        "MEDIUM" if overall_conf > 0.50 and any_detected else
        "LOW"    if any_detected else "NONE"
    )

    return {
        "is_stego"          : any_detected,
        "overall_risk"      : overall_risk,
        "overall_confidence": round(overall_conf, 3),
        "findings"          : findings,
        "image_stats"       : {
            "width"       : w,
            "height"      : h,
            "mode"        : mode,
            "file_size_kb": round(file_kb, 1),
            "entropy"     : round(entropy, 3),
        },
        "duration_sec": 0.6,
    }
    # ── END STUB ──────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_stego():
    st.sidebar.markdown("""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: #4B5563;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    ">STEGO · DETECTION OPTIONS</div>
    """, unsafe_allow_html=True)

    selected_labels = st.sidebar.multiselect(
        "Detection Techniques",
        options=list(STEGO_TECHNIQUES.keys()),
        default=list(STEGO_TECHNIQUES.keys())[:4],
        help="Select which steganography techniques to run. More = slower but thorough.",
    )
    techniques = [STEGO_TECHNIQUES[l] for l in selected_labels]

    st.sidebar.markdown("---")

    show_channels = st.sidebar.toggle("Show Channel Inspector", value=True,
        help="Display individual R/G/B and LSB planes for visual inspection")
    show_histogram = st.sidebar.toggle("Show Entropy Histogram", value=False,
        help="Plot pixel value distribution — uniform dist can indicate encryption")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.65rem;color:#374151;line-height:1.7;font-family:'JetBrains Mono',monospace;">
    <b style="color:#4B5563;">DETECTION METHODS</b><br>
    LSB · statistical signature<br>
    DCT · JPEG coefficient bias<br>
    EXIF · metadata embedding<br>
    Chi-Square · pixel uniformity<br>
    RS · Regular-Singular pairs
    </div>
    """, unsafe_allow_html=True)

    return techniques, show_channels, show_histogram


# ─────────────────────────────────────────────────────────────────────────────
# RESULT RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_stego_banner(result: dict):
    is_stego = result.get("is_stego", False)
    risk     = result.get("overall_risk", "NONE")
    conf     = result.get("overall_confidence", 0.0)
    fg, bg   = RISK_COLORS.get(risk, ("#8B9AC7", "#151B2D"))
    icon     = RISK_ICONS.get(risk, "⚪")

    if not is_stego:
        label = "CLEAN — No steganographic content detected"
        sublabel = "No hidden payload indicators found across selected techniques."
    else:
        n = sum(1 for f in result.get("findings", []) if f.get("detected"))
        label = f"{risk} — {n} technique{'s' if n != 1 else ''} flagged hidden content"
        sublabel = f"Overall confidence: {conf:.0%} · Immediate manual inspection recommended."

    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid {fg}33;
        border-left: 3px solid {fg};
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                color: {fg};
            ">{icon} {label}</div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: {fg}99;
                margin-top: 4px;
            ">{sublabel}</div>
        </div>
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: {fg}44;
        ">{risk}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_technique_results(findings: list):
    if not findings:
        st.info("No findings to display.")
        return

    rows_html = ""
    for f in findings:
        detected = f.get("detected", False)
        risk     = f.get("risk_level", "NONE")
        fg, _    = RISK_COLORS.get(risk if detected else "NONE", ("#8B9AC7", "#151B2D"))
        icon     = RISK_ICONS.get(risk if detected else "NONE", "✅")
        conf     = f.get("confidence", 0.0)
        bar_w    = int(conf * 100)
        label    = f.get("label", f.get("technique", "?"))
        detail   = f.get("detail", "")
        payload  = f.get("payload_hint")

        payload_html = f"""
        <span style="
            background: #FF3B3B22;
            color: #FF3B3B;
            border: 1px solid #FF3B3B44;
            border-radius: 3px;
            padding: 1px 7px;
            font-size: 0.63rem;
            font-weight: 600;
            margin-left: 8px;
        ">📦 {payload}</span>""" if payload else ""

        rows_html += f"""
        <div style="
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin-bottom: 8px;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        ">
            <div style="
                width: 18px; height: 18px;
                border-radius: 50%;
                background: {fg}22;
                border: 1px solid {fg};
                display: flex; align-items: center; justify-content: center;
                font-size: 0.65rem;
                flex-shrink: 0;
                margin-top: 2px;
            ">{icon}</div>
            <div style="flex: 1; min-width: 0;">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 4px;
                ">
                    <span style="
                        font-size: 0.78rem;
                        font-weight: 600;
                        color: {'#E2E8F0' if detected else '#4B5563'};
                        letter-spacing: 0.03em;
                    ">{label}</span>
                    {payload_html}
                </div>
                <div style="
                    font-size: 0.68rem;
                    color: #6B7280;
                    line-height: 1.5;
                    margin-bottom: 6px;
                ">{detail}</div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="
                        width: 120px; height: 4px;
                        background: #1E293B; border-radius: 2px; overflow: hidden;
                    ">
                        <div style="
                            width: {bar_w}%; height: 100%;
                            background: {fg}; border-radius: 2px;
                        "></div>
                    </div>
                    <span style="font-size:0.68rem;color:#8B9AC7;">{conf:.0%} confidence</span>
                </div>
            </div>
        </div>
        """

    st.markdown(rows_html, unsafe_allow_html=True)


def _render_image_stats(stats: dict, entropy: float):
    hi_entropy = entropy > 7.5
    entropy_color = "#FF8C42" if hi_entropy else "#06D6A0"
    entropy_note  = "HIGH — may indicate encryption or compressed payload" if hi_entropy else "NORMAL"

    st.markdown(f"""
    <div style="
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1rem 1.25rem;
    ">
        <div style="
            font-size: 0.65rem; font-weight: 600; color: #4B5563;
            letter-spacing: 0.15em; text-transform: uppercase;
            margin-bottom: 0.85rem;
        ">IMAGE STATISTICS</div>
        <div style="display:flex; flex-wrap:wrap; gap:1.5rem;">
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px;">DIMENSIONS</div>
                <div style="font-size:0.82rem;font-weight:600;color:#CBD5E1;">{stats.get('width','?')} × {stats.get('height','?')}</div>
            </div>
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px;">COLOUR MODE</div>
                <div style="font-size:0.82rem;font-weight:600;color:#CBD5E1;">{stats.get('mode','?')}</div>
            </div>
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px;">FILE SIZE</div>
                <div style="font-size:0.82rem;font-weight:600;color:#CBD5E1;">{stats.get('file_size_kb','?')} KB</div>
            </div>
            <div>
                <div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px;">ENTROPY</div>
                <div style="font-size:0.82rem;font-weight:600;color:{entropy_color};">{entropy:.3f} bits <span style="font-size:0.63rem;color:{entropy_color}99;">({entropy_note})</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_channel_inspector(image_path: str):
    """
    Show individual R, G, B channels and the LSB plane.
    Pure OpenCV/numpy — no external stego module needed.
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            st.warning("Could not load image for channel inspection.")
            return

        b, g, r = cv2.split(img)
        lsb_plane = (r & 1) * 255   # amplify LSB to visible range

        cols = st.columns(4)
        for col, (channel, name, tint) in zip(
            cols,
            [(r, "RED", [0,0,255]),
             (g, "GREEN", [0,255,0]),
             (b, "BLUE", [255,0,0]),
             (lsb_plane, "LSB PLANE", [255,255,0])]
        ):
            with col:
                # Render as greyscale image with a coloured border
                col.markdown(f"""
                <div style="
                    font-size:0.6rem;color:#4B5563;
                    letter-spacing:0.12em;text-transform:uppercase;
                    text-align:center;margin-bottom:4px;
                ">{name}</div>""", unsafe_allow_html=True)
                col.image(channel, use_container_width=True, clamp=True)

        st.markdown("""
        <div style="font-size:0.65rem;color:#374151;margin-top:4px;font-family:'JetBrains Mono',monospace;">
        LSB PLANE: amplified least-significant bits.
        Visible patterns (faces, text) indicate hidden data. Noise = clean image.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Channel inspector error: {e}")


def _render_entropy_histogram(image_path: str):
    try:
        import cv2
        import matplotlib.pyplot as plt

        img  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()

        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#0A0D14")
        ax.fill_between(range(256), hist, alpha=0.7, color="#00AEEF")
        ax.plot(range(256), hist, color="#00AEEF", linewidth=0.8)
        ax.set_xlim(0, 255)
        ax.tick_params(colors="#4B5563", labelsize=7)
        ax.spines[:].set_color("#1E293B")
        ax.set_ylabel("Frequency", color="#4B5563", fontsize=7)
        ax.set_xlabel("Pixel Value", color="#4B5563", fontsize=7)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Histogram error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TAB RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab_stego():
    techniques, show_channels, show_histogram = render_sidebar_stego()

    # ── Upload zone ────────────────────────────────────────────────────────────
    col_up, col_btn = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drop image here or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "webp"],
            label_visibility="collapsed",
            key="stego_uploader",
        )
    with col_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("⟶  ANALYSE", use_container_width=True, key="stego_scan")

    if not techniques:
        st.warning("Select at least one detection technique in the sidebar.")
        return

    if not uploaded:
        render_empty_state("Steganography")
        return

    # ── Run analysis ───────────────────────────────────────────────────────────
    result   = st.session_state.get("stego_result")
    last_file= st.session_state.get("stego_last_file")
    tmp_path = st.session_state.get("stego_tmp_path")

    if scan_clicked or result is None or last_file != uploaded.name:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded.name).suffix,
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Running steganography analysis…"):
                result = _run_stego_analysis(tmp_path, techniques)

            st.session_state["stego_result"]    = result
            st.session_state["stego_last_file"] = uploaded.name
            st.session_state["stego_tmp_path"]  = tmp_path

        except Exception as e:
            st.error(f"Analysis error: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            return

    if result is None:
        return

    # ── Results dashboard ──────────────────────────────────────────────────────
    _render_stego_banner(result)

    col_img, col_findings = st.columns([1, 1], gap="large")

    with col_img:
        section_label("ORIGINAL IMAGE")
        if tmp_path and os.path.exists(tmp_path):
            st.image(tmp_path, use_container_width=True)
        else:
            st.info("Image not available for preview.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        stats   = result.get("image_stats", {})
        entropy = stats.get("entropy", 0.0)
        _render_image_stats(stats, entropy)

        if show_histogram:
            section_label("PIXEL DISTRIBUTION", color="#00AEEF")
            if tmp_path and os.path.exists(tmp_path):
                _render_entropy_histogram(tmp_path)

    with col_findings:
        section_label("TECHNIQUE RESULTS")
        _render_technique_results(result.get("findings", []))

    if show_channels and tmp_path and os.path.exists(tmp_path):
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section_label("CHANNEL INSPECTOR", color="#00AEEF")
        _render_channel_inspector(tmp_path)

    # Duration
    duration = result.get("duration_sec", 0)
    st.markdown(f"""
    <div style="
        text-align: right;
        font-size: 0.65rem;
        color: #374151;
        margin-top: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    ">Analysis completed in {duration:.2f}s · {uploaded.name}</div>
    """, unsafe_allow_html=True)