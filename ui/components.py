"""
ui/components.py
─────────────────
Shared styled components for the Barclays Image DLP Platform UI.

All components return HTML strings rendered via st.markdown(... unsafe_allow_html=True),
or use native Streamlit widgets styled via the theme.
"""

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "CRITICAL": ("#FF3B3B", "#2D0A0A"),
    "HIGH"    : ("#FF8C42", "#2D1A0A"),
    "MEDIUM"  : ("#FFD166", "#2D270A"),
    "LOW"     : ("#06D6A0", "#0A2D25"),
    "REVIEW"  : ("#8B9AC7", "#151B2D"),
    "NONE"    : ("#06D6A0", "#0A2D25"),
}

RISK_ICONS = {
    "CRITICAL": "🔴",
    "HIGH"    : "🟠",
    "MEDIUM"  : "🟡",
    "LOW"     : "🟢",
    "REVIEW"  : "⚪",
    "NONE"    : "✅",
}

SOURCE_BADGE = {
    "regex"  : ("#00AEEF", "#001D2D"),
    "nlp"    : ("#A855F7", "#1A0D2D"),
    "vision" : ("#F59E0B", "#2D1F0A"),
}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def inject_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Root overrides ── */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #0A0D14; }
    ::-webkit-scrollbar-thumb { background: #00AEEF44; border-radius: 2px; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #111827;
        border-radius: 0;
        border-bottom: 1px solid #1E293B;
        padding: 0 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4B5563;
        padding: 0.85rem 1.5rem;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #00AEEF !important;
        border-bottom: 2px solid #00AEEF !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #1E293B;
        border-radius: 8px;
        background: #111827;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00AEEF44;
    }

    /* ── Dividers ── */
    hr { border-color: #1E293B !important; margin: 1.25rem 0; }

    /* ── Button overrides ── */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 4px;
        border: 1px solid #00AEEF;
        background: transparent;
        color: #00AEEF;
        padding: 0.5rem 1.5rem;
        transition: all 0.15s;
    }
    .stButton > button:hover {
        background: #00AEEF;
        color: #0A0D14;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }

    /* ── Select / radio ── */
    .stSelectbox > div > div, .stRadio > div {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #8B9AC7;
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0 1.25rem 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 0.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="
                width: 36px; height: 36px;
                background: #00AEEF;
                border-radius: 4px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.1rem;
            ">🔍</div>
            <div>
                <div style="
                    font-family: 'Syne', sans-serif;
                    font-size: 1.25rem;
                    font-weight: 800;
                    color: #E2E8F0;
                    letter-spacing: -0.01em;
                    line-height: 1;
                ">IMAGE DLP PLATFORM</div>
                <div style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.65rem;
                    color: #4B5563;
                    letter-spacing: 0.15em;
                    text-transform: uppercase;
                    margin-top: 3px;
                ">Data Loss Prevention · Barclays Security</div>
            </div>
        </div>
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #4B5563;
            letter-spacing: 0.08em;
            text-align: right;
        ">
            <div style="color: #06D6A0; margin-bottom: 2px;">● SYSTEM ONLINE</div>
            <div>OCR · NLP · VISION · STEGO</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RISK BANNER
# ─────────────────────────────────────────────────────────────────────────────

def render_risk_banner(risk_level: str, total_findings: int, is_sensitive: bool):
    fg, bg = RISK_COLORS.get(risk_level, ("#8B9AC7", "#151B2D"))
    icon = RISK_ICONS.get(risk_level, "⚪")

    if not is_sensitive:
        label = "CLEAN — No sensitive content detected"
        sublabel = "All checks passed. Image cleared."
        risk_level = "NONE"
        fg, bg = RISK_COLORS["NONE"]
    else:
        label = f"{risk_level} SENSITIVITY — {total_findings} finding{'s' if total_findings != 1 else ''} detected"
        sublabel = {
            "CRITICAL": "Immediate review required. Do not share this image.",
            "HIGH"    : "Very likely sensitive. Analyst review recommended.",
            "MEDIUM"  : "Probably sensitive. Prioritise for review.",
            "LOW"     : "Possible sensitive content. Low priority review.",
            "REVIEW"  : "Flagged for review. Possible false positive.",
        }.get(risk_level, "")

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
                letter-spacing: 0.02em;
            ">{icon} {label}</div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: {fg}99;
                margin-top: 4px;
                letter-spacing: 0.05em;
            ">{sublabel}</div>
        </div>
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: {fg}44;
            letter-spacing: -0.02em;
        ">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FINDINGS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def render_findings_table(matches: list):
    if not matches:
        st.markdown(
            '<div style="padding:2rem;text-align:center;color:#4B5563;font-size:0.8rem;'
            'border:1px dashed #1E293B;border-radius:6px;">No findings to display</div>',
            unsafe_allow_html=True,
        )
        return

    # Column header row
    st.markdown(
        '<div style="display:grid;grid-template-columns:90px 1fr 120px 80px;'
        'gap:0;background:#0D1117;border:1px solid #1E293B;border-radius:6px 6px 0 0;'
        'padding:0.4rem 0.75rem;">'
        '<span style="font-size:0.6rem;font-weight:600;color:#4B5563;letter-spacing:0.12em;text-transform:uppercase;">RISK</span>'
        '<span style="font-size:0.6rem;font-weight:600;color:#4B5563;letter-spacing:0.12em;text-transform:uppercase;">TYPE</span>'
        '<span style="font-size:0.6rem;font-weight:600;color:#4B5563;letter-spacing:0.12em;text-transform:uppercase;">CONFIDENCE</span>'
        '<span style="font-size:0.6rem;font-weight:600;color:#4B5563;letter-spacing:0.12em;text-transform:uppercase;">SOURCE</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, m in enumerate(matches):
        risk    = m.get("risk_level", "REVIEW")
        fg, _   = RISK_COLORS.get(risk, ("#8B9AC7", "#151B2D"))
        icon    = RISK_ICONS.get(risk, "⚪")
        score   = m.get("unified_score", 0.0)
        ftype   = m.get("type", "unknown").replace("_", " ").upper()
        source  = m.get("source", "regex")
        src_fg, src_bg = SOURCE_BADGE.get(source, ("#8B9AC7", "#151B2D"))
        bar_w   = int(score * 100)
        is_last = i == len(matches) - 1
        border_radius = "0 0 6px 6px" if is_last else "0"

        tags = ""
        if m.get("fp_risk"):
            tags += '<span style="color:#FFD166;font-size:0.6rem;margin-left:6px;font-family:JetBrains Mono,monospace;">[FP?]</span>'
        if m.get("vision_solo"):
            tags += '<span style="color:#F59E0B;font-size:0.6rem;margin-left:6px;font-family:JetBrains Mono,monospace;">[vision]</span>'

        row_html = (
            f'<div style="display:grid;grid-template-columns:90px 1fr 120px 80px;'
            f'gap:0;background:#111827;border-left:1px solid #1E293B;border-right:1px solid #1E293B;'
            f'border-bottom:1px solid #1E293B;border-radius:{border_radius};'
            f'padding:0.55rem 0.75rem;align-items:center;">'

            # RISK cell
            f'<div style="white-space:nowrap;">'
            f'<span style="font-size:0.8rem;">{icon}</span>'
            f'<span style="font-size:0.68rem;font-weight:600;color:{fg};'
            f'margin-left:5px;letter-spacing:0.06em;font-family:JetBrains Mono,monospace;">{risk}</span>'
            f'</div>'

            # TYPE cell
            f'<div>'
            f'<span style="font-size:0.72rem;color:#CBD5E1;letter-spacing:0.04em;'
            f'font-family:JetBrains Mono,monospace;">{ftype}</span>'
            f'{tags}'
            f'</div>'

            # CONFIDENCE cell
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:60px;height:4px;background:#1E293B;border-radius:2px;overflow:hidden;">'
            f'<div style="width:{bar_w}%;height:100%;background:{fg};border-radius:2px;"></div>'
            f'</div>'
            f'<span style="font-size:0.68rem;color:#8B9AC7;font-weight:600;'
            f'font-family:JetBrains Mono,monospace;">{score:.3f}</span>'
            f'</div>'

            # SOURCE cell
            f'<div>'
            f'<span style="background:{src_bg};color:{src_fg};border:1px solid {src_fg}55;'
            f'border-radius:3px;padding:2px 7px;font-size:0.6rem;font-weight:600;'
            f'letter-spacing:0.08em;text-transform:uppercase;'
            f'font-family:JetBrains Mono,monospace;">{source}</span>'
            f'</div>'

            f'</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE BREAKDOWN CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_score_breakdown(scored_result: dict):
    summary   = scored_result.get("score_summary", {})
    ocr_q     = scored_result.get("ocr_quality", "—")
    total     = scored_result.get("total", 0)
    nlp_avail = scored_result.get("nlp_available", False)
    ocr_color = {"good": "#06D6A0", "moderate": "#FFD166", "poor": "#FF8C42"}.get(ocr_q, "#8B9AC7")
    nlp_color = "#06D6A0" if nlp_avail else "#FFD166"
    nlp_label = "ACTIVE" if nlp_avail else "PARTIAL"

    # Header
    st.markdown(
        '<div style="background:#111827;border:1px solid #1E293B;border-radius:6px 6px 0 0;'
        'padding:0.6rem 1rem 0.4rem 1rem;">'
        '<span style="font-size:0.65rem;font-weight:600;color:#4B5563;'
        'letter-spacing:0.15em;text-transform:uppercase;">SCORE DISTRIBUTION</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "REVIEW"]
    has_any = any(summary.get(lvl, 0) > 0 for lvl in order)

    if not has_any:
        st.markdown(
            '<div style="background:#111827;border-left:1px solid #1E293B;'
            'border-right:1px solid #1E293B;border-bottom:1px solid #1E293B;'
            'padding:0.6rem 1rem;color:#4B5563;font-size:0.75rem;'
            'font-family:JetBrains Mono,monospace;">No findings scored</div>',
            unsafe_allow_html=True,
        )
    else:
        for level in order:
            count = summary.get(level, 0)
            if count == 0:
                continue
            fg, _ = RISK_COLORS.get(level, ("#8B9AC7", "#151B2D"))
            pct   = int((count / max(1, total)) * 100)
            st.markdown(
                f'<div style="background:#111827;border-left:1px solid #1E293B;'
                f'border-right:1px solid #1E293B;border-bottom:1px solid #1E293B;'
                f'padding:0.4rem 1rem;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<div style="width:60px;font-size:0.65rem;font-weight:600;color:{fg};'
                f'letter-spacing:0.08em;font-family:JetBrains Mono,monospace;">{level}</div>'
                f'<div style="flex:1;height:5px;background:#1E293B;border-radius:3px;overflow:hidden;">'
                f'<div style="width:{pct}%;height:100%;background:{fg};border-radius:3px;"></div>'
                f'</div>'
                f'<div style="width:20px;text-align:right;font-size:0.68rem;color:#8B9AC7;'
                f'font-weight:600;font-family:JetBrains Mono,monospace;">{count}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Footer stats row
    st.markdown(
        f'<div style="background:#111827;border-left:1px solid #1E293B;'
        f'border-right:1px solid #1E293B;border-bottom:1px solid #1E293B;'
        f'border-radius:0 0 6px 6px;padding:0.6rem 1rem;'
        f'border-top:1px solid #1E293B;display:flex;gap:1.5rem;">'
        f'<div>'
        f'<div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:3px;font-family:JetBrains Mono,monospace;">OCR QUALITY</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:{ocr_color};'
        f'font-family:JetBrains Mono,monospace;">{ocr_q.upper()}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:3px;font-family:JetBrains Mono,monospace;">NLP ENGINE</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:{nlp_color};'
        f'font-family:JetBrains Mono,monospace;">{nlp_label}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:0.6rem;color:#4B5563;letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:3px;font-family:JetBrains Mono,monospace;">TOTAL FINDINGS</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:#E2E8F0;'
        f'font-family:JetBrains Mono,monospace;">{total}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGE TIMELINE (shows which stages ran + results)
# ─────────────────────────────────────────────────────────────────────────────

def render_pipeline_stages(result: dict):
    scored  = result.get("scored_result", {})
    vision  = result.get("vision_result", {})
    regex_r = result.get("regex_result", {})
    nlp_r   = result.get("nlp_result", {})

    stages = [
        ("PREPROCESS", "✓", "#06D6A0",
            "Image cleaned + scaled"),
        ("OCR", "✓", "#06D6A0",
            f"{scored.get('_ocr_word_count', '?')} words extracted · quality: {scored.get('ocr_quality', '?')}"),
        ("REGEX DLP",
            "✓" if regex_r.get("total", 0) > 0 else "–",
            "#06D6A0" if regex_r.get("total", 0) > 0 else "#4B5563",
            f"{regex_r.get('total', 0)} pattern match(es)"),
        ("NLP / NER", "✓", "#06D6A0",
            f"+{len(nlp_r.get('new_findings', []))} findings · {len(nlp_r.get('context_flags', []))} FP flags"),
        ("CONFIDENCE", "✓", "#06D6A0",
            f"overall_risk={scored.get('overall_risk', '?')}"),
        ("VISION", "✓", "#06D6A0",
            f"{vision.get('document_type', '?')} · conf={vision.get('type_confidence', 0):.2f} · {vision.get('model_used', '?')}"),
    ]

    # Header
    st.markdown(
        '<div style="background:#111827;border:1px solid #1E293B;border-radius:6px 6px 0 0;'
        'padding:0.6rem 1rem 0.4rem 1rem;">'
        '<span style="font-size:0.65rem;font-weight:600;color:#4B5563;'
        'letter-spacing:0.15em;text-transform:uppercase;">PIPELINE STAGES</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, (name, status, color, detail) in enumerate(stages):
        is_last = i == len(stages) - 1
        br = "0 0 6px 6px" if is_last else "0"
        connector = (
            "" if is_last else
            '<div style="width:1px;height:10px;background:#1E293B;margin:2px 0 2px 8px;"></div>'
        )
        st.markdown(
            f'<div style="background:#111827;border-left:1px solid #1E293B;'
            f'border-right:1px solid #1E293B;border-bottom:1px solid #1E293B;'
            f'border-radius:{br};padding:0.45rem 1rem;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="width:16px;height:16px;border-radius:50%;background:{color}22;'
            f'border:1px solid {color};display:flex;align-items:center;justify-content:center;'
            f'font-size:0.55rem;color:{color};font-weight:700;flex-shrink:0;">{status}</div>'
            f'<div>'
            f'<div style="font-size:0.68rem;font-weight:600;color:#CBD5E1;'
            f'letter-spacing:0.06em;font-family:JetBrains Mono,monospace;">{name}</div>'
            f'<div style="font-size:0.6rem;color:#4B5563;font-family:JetBrains Mono,monospace;">{detail}</div>'
            f'</div></div>'
            f'{connector}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION LABEL
# ─────────────────────────────────────────────────────────────────────────────

def section_label(text: str, color: str = "#4B5563"):
    st.markdown(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: {color};
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    ">{text}</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

def render_empty_state(tab_name: str = "DLP"):
    st.markdown(f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        border: 1px dashed #1E293B;
        border-radius: 8px;
        background: #111827;
        margin-top: 1rem;
        text-align: center;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.4;">📁</div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 0.95rem;
            font-weight: 700;
            color: #4B5563;
            letter-spacing: 0.02em;
            margin-bottom: 0.5rem;
        ">Upload an image to begin {tab_name} analysis</div>
        <div style="
            font-size: 0.72rem;
            color: #374151;
            max-width: 320px;
            line-height: 1.6;
        ">Supported formats: JPG · PNG · BMP · TIFF<br>Configure options in the sidebar before scanning.</div>
    </div>
    """, unsafe_allow_html=True)