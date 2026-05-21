"""
app.py — Barclays Image DLP Platform
──────────────────────────────────────
Streamlit entrypoint.  Run with:

    streamlit run app.py

Architecture
────────────
This file owns:
  • Page config (title, icon, layout)
  • Global CSS injection
  • Top header bar
  • Tab routing — DLP tab → ui/tab_dlp.py
                  Stego tab → ui/tab_stego.py
  • Sidebar tab-switching (sidebar content is
    rendered by each tab module itself)

Everything else (pipeline wiring, result rendering,
sidebar options) lives in the tab modules.
"""

import sys
import os
from pathlib import Path

# ── Make sure project root is importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Must be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title  = "Image DLP Platform — Barclays",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

from ui.components import inject_global_css, render_header
from ui.tab_dlp    import render_tab_dlp
from ui.tab_stego  import render_tab_stego


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES + HEADER
# ─────────────────────────────────────────────────────────────────────────────

inject_global_css()
render_header()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_dlp, tab_stego = st.tabs([
    "🔍  OCR / NLP / VISION DLP",
    "🕵️  STEGANOGRAPHY DETECTION",
])

with tab_dlp:
    render_tab_dlp()

with tab_stego:
    render_tab_stego()