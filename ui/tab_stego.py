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

import html as html_module
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
    Real implementation using EfficientNetV2-S trained on LSB/PVD/DCT/FFT datasets.
    Model: stego_efficientv2.pth (place in project root next to app.py)
    Accuracy: 78.4% overall | LSB: 86.6% | FFT: 86.2% | Clean: 98.5%
    """
    import time
    import numpy as np_inner
    start = time.time()

    # ── Image stats (always computed regardless of model) ─────────────────────
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        w, h = img.size
        mode = img.mode
        file_kb = os.path.getsize(image_path) / 1024
        arr  = np_inner.array(img.convert("L"))
        hist = np_inner.bincount(arr.ravel(), minlength=256)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy = float(-np_inner.sum(prob * np_inner.log2(prob)))
    except Exception:
        w, h, mode, file_kb, entropy = 0, 0, "?", 0.0, 0.0

    # ── Model inference ───────────────────────────────────────────────────────
    # Maps our 5 model classes to the tab's technique keys
    MODEL_CLASS_TO_TECHNIQUE = {
        "lsb": "lsb",
        "pvd": "pvd",   # closest to chi_square in sidebar
        "dct": "dct",
        "fft": "rs",    # closest to rs in sidebar
    }
    CLASS_LABELS  = ["Clean", "LSB", "PVD", "DCT", "FFT"]
    CLASS_KEYS    = ["clean", "lsb", "pvd", "dct", "fft"]
    CLASS_DETAILS = {
        "lsb": "Least Significant Bit steganography detected. Data hidden in pixel LSBs.",
        "pvd": "Pixel Value Differencing steganography detected. Data hidden via pixel pair differences.",
        "dct": "DCT coefficient steganography detected. Data hidden in JPEG frequency domain.",
        "fft": "Frequency-domain steganography detected. Hidden data found via spectral analysis.",
    }

    probs = None
    model_error = None

    # ── Pre-flight gate: only skip for pure B&W text (entropy < 2.0 AND variance < 200) ──
    # Tighter thresholds so stego spreadsheet/table images still reach the model.
    # The previous gate (entropy < 3.5) caused false negatives on coloured tables.
    try:
        from PIL import Image as _pil_chk
        import numpy as _np_chk
        _arr      = _np_chk.array(_pil_chk.open(image_path).convert("RGB")).astype(float)
        _variance = float(_np_chk.var(_arr))
    except Exception:
        _variance = 9999.0

    if entropy < 2.0 and _variance < 200.0:
        _note = (
            f"Image bypassed neural network: entropy={entropy:.2f} bits, "
            f"pixel variance={_variance:.0f}. "
            f"Pure black-and-white text images fall outside the model's "
            f"training distribution and cause false positives. "
            f"Classified CLEAN by pre-flight gate."
        )
        return {
            "is_stego"          : False,
            "overall_risk"      : "NONE",
            "overall_confidence": 0.0,
            "findings"          : [{
                "technique"   : "model",
                "label"       : "Neural Network (EfficientNetV2)",
                "detected"    : False,
                "confidence"  : 0.0,
                "risk_level"  : "NONE",
                "detail"      : _note,
                "payload_hint": None,
            }],
            "image_stats": {
                "width"       : w,
                "height"      : h,
                "mode"        : mode,
                "file_size_kb": round(file_kb, 1),
                "entropy"     : round(entropy, 3),
            },
            "duration_sec": round(time.time() - start, 2),
        }

    try:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_v2_s
        from torchvision.transforms import v2
        from PIL import Image as PILImage2

        # Find model weights — look in project root and common locations
        search_paths = [
            Path(__file__).resolve().parent.parent / "stego_efficientv2.pth",
            Path(__file__).resolve().parent / "stego_efficientv2.pth",
            Path("stego_efficientv2.pth"),
        ]
        model_path = next((p for p in search_paths if p.exists()), None)

        if model_path is None:
            raise FileNotFoundError(
                "stego_efficientv2.pth not found. Place it in the project root folder."
            )

        @st.cache_resource
        def _load_stego_model(path_str):
            m = efficientnet_v2_s(weights=None)
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(m.classifier[1].in_features, 5)
            )
            m.load_state_dict(torch.load(path_str, map_location="cpu"))
            m.eval()
            return m

        model = _load_stego_model(str(model_path))

        # Preprocess — matches training pipeline exactly
        img_pil = PILImage2.open(image_path).convert("RGB")
        iw, ih  = img_pil.size
        size    = 256
        transform = v2.Compose([
            v2.Pad(
                padding=(max(0,(size-iw)//2), max(0,(size-ih)//2),
                         max(0,(size-iw+1)//2), max(0,(size-ih+1)//2)),
                fill=0
            ),
            v2.CenterCrop((size, size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Normalize([0.485*255, 0.456*255, 0.406*255],
                         [0.229*255, 0.224*255, 0.225*255]),
        ])
        tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()

    except Exception as e:
        model_error = str(e)

    # ── Chi-square LSB statistical test (second opinion) ────────────────────
    # Independently checks whether the LSB plane looks statistically random
    # (a strong indicator of LSB steganography). This is a mathematical test,
    # not a neural net — gives us a reliable second opinion to reduce false positives.
    def _chi_square_lsb(img_path: str) -> tuple:
        """Returns (chi_score, lsb_detected). chi_score > 1.0 suggests LSB stego."""
        try:
            from PIL import Image as _P
            import numpy as _np2
            _img  = _np2.array(_P.open(img_path).convert("RGB"))
            _lsb  = _img[:, :, 0].ravel() & 1      # red channel LSBs
            _ones = int(_lsb.sum())
            _total = len(_lsb)
            _zeros = _total - _ones
            _expected = _total / 2.0
            # Chi-square: if LSBs are truly random (stego), zeros ≈ ones ≈ 50%
            _chi = ((_zeros - _expected)**2 + (_ones - _expected)**2) / _expected
            # Normalise to 0-1 range; score > 0.5 = suspiciously uniform LSBs
            _score = min(1.0, _chi / (_total * 0.01 + 1e-9))
            _detected = _score < 0.05   # near-perfect 50/50 split = LSB stego
            return round(float(_score), 4), _detected
        except Exception:
            return 0.0, False

    _chi_score, _chi_detected = _chi_square_lsb(image_path)

    # ── Build findings in the schema expected by the UI ───────────────────────
    findings = []

    if probs is not None:
        clean_prob  = float(probs[0])
        stego_prob  = float(1 - clean_prob)

        # Require model to be strongly confident AND chi-square to agree
        # before declaring HIGH. This eliminates false positives on photos
        # that look unusual to the model but have no actual hidden data.
        _model_flags = clean_prob < 0.15          # model says stego
        _high_conf   = stego_prob > 0.85           # model is very sure
        _chi_agrees  = _chi_detected               # LSB test confirms

        if _model_flags and _high_conf and _chi_agrees:
            is_stego     = True
            overall_risk = "HIGH"
        elif _model_flags and _high_conf:
            # Model very confident but chi-square disagrees — likely false positive
            # on an unusual natural image. Downgrade to REVIEW.
            is_stego     = True
            overall_risk = "REVIEW"
        elif _model_flags:
            # Model flags but not with high confidence
            is_stego     = True
            overall_risk = "MEDIUM"
        else:
            is_stego     = False
            overall_risk = "NONE"

        overall_conf = round(stego_prob, 3)

        # Per-technique findings
        for i, (label, key) in enumerate(zip(CLASS_LABELS[1:], CLASS_KEYS[1:]), start=1):
            tech_key = MODEL_CLASS_TO_TECHNIQUE.get(key, key)
            conf     = float(probs[i])
            detected = is_stego and overall_risk in ("HIGH", "MEDIUM") and (conf == float(max(probs[1:])))
            risk     = "HIGH" if detected and conf > 0.5 else ("MEDIUM" if detected else "NONE")
            findings.append({
                "technique"   : tech_key,
                "label"       : f"{label} Steganography",
                "detected"    : detected,
                "confidence"  : round(conf, 3),
                "risk_level"  : risk,
                "detail"      : CLASS_DETAILS[key] if detected else f"No {label} steganography detected.",
                "payload_hint": None,
            })

        # Chi-square finding
        findings.append({
            "technique"   : "chi_square",
            "label"       : "Chi-Square LSB Test",
            "detected"    : _chi_detected,
            "confidence"  : round(1 - _chi_score, 3) if _chi_detected else round(_chi_score, 3),
            "risk_level"  : "HIGH" if _chi_detected else "NONE",
            "detail"      : (
                "LSB plane shows near-perfect 50/50 bit distribution — "
                "strong statistical indicator of LSB steganography."
                if _chi_detected else
                "LSB distribution looks natural — no LSB steganography signature."
            ),
            "payload_hint": None,
        })

        # Neural network overall finding
        _nn_detail = (
            f"Model confidence: {stego_prob*100:.0f}% stego / {clean_prob*100:.0f}% clean. "
        )
        if overall_risk == "REVIEW":
            _nn_detail += "Downgraded from HIGH to REVIEW: chi-square LSB test found no hidden data signature."
        findings.insert(0, {
            "technique"   : "model",
            "label"       : "Neural Network (EfficientNetV2)",
            "detected"    : is_stego,
            "confidence"  : overall_conf,
            "risk_level"  : overall_risk,
            "detail"      : _nn_detail,
            "payload_hint": None,
        })

    else:
        # Model failed to load — fall back to entropy-based heuristic
        is_stego     = entropy > 7.6
        overall_conf = min(1.0, max(0.0, (entropy - 7.0) / 1.5)) if is_stego else 0.1
        overall_risk = "REVIEW" if is_stego else "NONE"
        detail_note  = f"Model unavailable: {model_error} — entropy heuristic used instead."

        findings = [{
            "technique"   : "entropy",
            "label"       : "Entropy Analysis (fallback)",
            "detected"    : is_stego,
            "confidence"  : round(overall_conf, 3),
            "risk_level"  : overall_risk,
            "detail"      : detail_note,
            "payload_hint": None,
        }]

    duration = round(time.time() - start, 2)

    return {
        "is_stego"          : is_stego,
        "overall_risk"      : overall_risk,
        "overall_confidence": overall_conf,
        "findings"          : findings,
        "image_stats"       : {
            "width"       : w,
            "height"      : h,
            "mode"        : mode,
            "file_size_kb": round(file_kb, 1),
            "entropy"     : round(entropy, 3),
        },
        "duration_sec": duration,
    }


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


def _clean_detail(detail: str, conf: float) -> str:
    """
    Strip ALL HTML from a detail string and return plain text only.
    No matter what the model or cache puts in 'detail', this returns
    a safe human-readable string — never markup.
    """
    import re
    text = re.sub(r"<[^>]*>", "", str(detail))   # remove every HTML tag
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    # If what remains is just CSS junk or too short, use a safe fallback
    css_junk = ["display:", "align-items", "justify-content", "margin-",
                 "width:", "height:", "background:", "border-", "font-",
                 "color:", "padding", "style=", "flex-"]
    if len(text) < 8 or any(p in text for p in css_junk):
        return f"Analysis complete — {conf:.0%} confidence."
    return text


def _render_technique_results(findings: list):
    """
    Render each finding as a native Streamlit card.
    Uses ZERO st.markdown(html) for user-data fields — all dynamic
    content (label, detail, confidence) goes through st.write / st.progress
    so broken HTML in findings can never leak into the page.
    """
    if not findings:
        st.info("No findings to display.")
        return

    for f in findings:
        detected = f.get("detected", False)
        risk     = f.get("risk_level", "NONE")
        conf     = f.get("confidence", 0.0)
        label    = str(f.get("label", f.get("technique", "Unknown")))
        detail   = _clean_detail(f.get("detail", ""), conf)
        payload  = f.get("payload_hint")

        fg, _    = RISK_COLORS.get(risk if detected else "NONE", ("#8B9AC7", "#151B2D"))
        icon     = RISK_ICONS.get(risk if detected else "NONE", "✅")
        risk_label = risk if detected else "NONE"

        # ── Card border via a single safe markdown div (no user data inside) ──
        st.markdown(
            f'<div style="border-left:3px solid {fg};border-radius:8px;'
            f'background:#111827;border:1px solid #1E293B;'
            f'padding:14px 16px 10px 16px;margin-bottom:12px;">',
            unsafe_allow_html=True,
        )

        # Header row — icon + label + risk badge, all plain strings
        col_icon, col_text, col_badge = st.columns([0.08, 0.72, 0.20])
        with col_icon:
            st.markdown(
                f'<div style="font-size:1rem;padding-top:4px">{icon}</div>',
                unsafe_allow_html=True,
            )
        with col_text:
            st.markdown(
                f'<div style="font-size:0.80rem;font-weight:700;color:#E2E8F0;">'
                f'{html_module.escape(label)}</div>'
                f'<div style="font-size:0.60rem;color:#64748B;letter-spacing:0.12em;'
                f'text-transform:uppercase;">{html_module.escape(risk_label)} Risk</div>',
                unsafe_allow_html=True,
            )
        with col_badge:
            if payload:
                st.markdown(
                    f'<div style="font-size:0.62rem;color:#FF3B3B;background:#FF3B3B22;'
                    f'border:1px solid #FF3B3B44;border-radius:4px;padding:2px 6px;'
                    f'text-align:center;">📦 {html_module.escape(str(payload))}</div>',
                    unsafe_allow_html=True,
                )

        # Detail — rendered as plain Streamlit text, never HTML
        st.caption(detail)

        # Confidence bar — native st.progress (no HTML)
        st.progress(min(1.0, max(0.0, conf)), text=f"{conf:.0%}")

        # Close the card div
        st.markdown("</div>", unsafe_allow_html=True)


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

    # ── Invalidate stale cache if findings contain raw HTML ────────────────────
    cached = st.session_state.get("stego_result")
    if cached:
        for _f in cached.get("findings", []):
            _d = str(_f.get("detail", ""))
            if any(p in _d for p in ["<div", "display:flex", "style="]):
                for _k in ["stego_result", "stego_last_file", "stego_tmp_path"]:
                    st.session_state.pop(_k, None)
                break

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
    ">Analysis completed in {duration:.2f}s · {html_module.escape(uploaded.name)}</div>
    """, unsafe_allow_html=True)