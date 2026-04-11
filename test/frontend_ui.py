# Frontend UI helpers (Streamlit layout, theme, and rendering components)

import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from pathlib import Path
from PIL import Image
from backend_portfolio import classify_risk, classify_esg


# ---------------------------------------------------------------------------
# Logo helpers
# ---------------------------------------------------------------------------

def _logo_path():
    """Return the absolute path to the full GreenGate logo file (icon + text)."""
    return Path(__file__).with_name("greengate_logo.png")


def _logo_b64():
    """Return base64-encoded full logo string for embedding in HTML, or None."""
    p = _logo_path()
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode("ascii")
    return None


def ensure_icon_cropped():
    """Crop the left 40% of greengate_logo.png to extract the gate/leaves icon
    only (no text). Always re-crops so the correct version is always on disk."""
    icon_path = os.path.join(os.path.dirname(__file__), "greengate_icon.png")
    full_path = os.path.join(os.path.dirname(__file__), "greengate_logo.png")
    orig = Image.open(full_path)
    w, h = orig.size
    icon = orig.crop((0, 0, int(w * 0.40), h)).convert("RGBA")
    data = np.array(icon)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    data[(r > 220) & (g > 220) & (b > 220)] = [0, 0, 0, 0]
    Image.fromarray(data).save(icon_path)
    return icon_path


def _icon_b64():
    """Return base64-encoded icon-only image for embedding in HTML, or None."""
    icon_path = Path(__file__).with_name("greengate_icon.png")
    if icon_path.exists():
        return base64.b64encode(icon_path.read_bytes()).decode("ascii")
    return None


# ---------------------------------------------------------------------------
# Sage contextual guide
# ---------------------------------------------------------------------------

def render_sage(message: str):
    """Render a subtle Sage contextual tip card."""
    st.markdown(
        f"""<div style="
            background:#f0faf4;
            border-left:3px solid #2d6a4f;
            padding:10px 14px;
            border-radius:4px;
            margin:8px 0 16px 0;
            font-size:13px;
            color:#2c3e35;
            font-style:italic;
        ">🌿 <strong>Sage</strong> &nbsp; {message}</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Landing / entry page
# ---------------------------------------------------------------------------

def render_landing_page():
    """Clean two-column landing page with ESG leaf animations."""

    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      /* ── Full-page white background ── */
      html, body, [data-testid="stAppViewContainer"],
      [data-testid="stMain"], .main, .block-container {
        background: #ffffff !important;
      }
      [data-testid="stHeader"],
      [data-testid="stToolbar"],
      footer { visibility: hidden !important; height: 0 !important; }

      @keyframes iconFloat {
        0%,100% { transform: translateY(0px) rotate(-1deg); }
        50%     { transform: translateY(-14px) rotate(1deg); }
      }
      @keyframes fadeSlideUp {
        0%   { opacity: 0; transform: translateY(22px); }
        100% { opacity: 1; transform: translateY(0);    }
      }
      @keyframes shimmer {
        0%,100% { background-position: 0% 50%; }
        50%     { background-position: 100% 50%; }
      }

      /* Landing content wrapper */
      .landing-wrap {
        position: relative;
        z-index: 1;
        min-height: 86vh;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeSlideUp 0.75s ease both;
      }
      .landing-inner {
        display: flex;
        align-items: center;
        gap: 3.5rem;
        max-width: 860px;
        width: 100%;
        padding: 2rem 1rem;
      }

      /* Icon */
      .landing-icon-col {
        flex: 0 0 240px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .landing-icon-col img {
        width: 240px;
        height: auto;
        animation: iconFloat 5s ease-in-out infinite;
        filter: drop-shadow(0 18px 44px rgba(26,92,42,0.22));
      }

      /* Text side */
      .landing-text-col {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }
      .landing-badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        background: #e8f5ee;
        color: #1a6b3c;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 0.85rem;
        width: fit-content;
        animation: fadeSlideUp 0.6s ease both 0.1s;
        opacity: 0;
      }
      .landing-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(3rem, 5.5vw, 3.8rem);
        font-weight: 800;
        color: #1a5c2a;
        letter-spacing: -0.05em;
        line-height: 1.0;
        margin: 0 0 0.5rem 0;
        animation: fadeSlideUp 0.65s ease both 0.2s;
        opacity: 0;
        background: linear-gradient(135deg, #1a5c2a 0%, #2d8a4e 60%, #4caf7d 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeSlideUp 0.65s ease both 0.2s, shimmer 4s ease infinite 1s;
      }
      .landing-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 500;
        color: #5a7a63;
        margin: 0 0 1.4rem 0;
        line-height: 1.55;
        animation: fadeSlideUp 0.65s ease both 0.3s;
        opacity: 0;
      }
      .landing-divider {
        width: 54px;
        height: 3px;
        background: linear-gradient(90deg, #1a5c2a, #4caf7d, #1a5c2a);
        background-size: 200% 100%;
        animation: fadeSlideUp 0.65s ease both 0.4s, shimmer 3s ease infinite;
        border-radius: 999px;
        margin-bottom: 1.1rem;
        opacity: 0;
      }
      .landing-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-style: italic;
        font-weight: 500;
        color: #1a6b3c;
        margin-bottom: 2rem;
        letter-spacing: 0.01em;
        animation: fadeSlideUp 0.65s ease both 0.5s;
        opacity: 0;
      }

      /* Enter button */
      div[data-testid="stButton"] > button {
        background: #2d6a4f !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 36px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 8px 28px rgba(45,106,79,0.30) !important;
        transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease !important;
        cursor: pointer;
      }
      div[data-testid="stButton"] > button:hover {
        background: #1b4332 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 36px rgba(27,67,50,0.38) !important;
      }
      div[data-testid="stButton"] > button p,
      div[data-testid="stButton"] > button span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # Build icon HTML using icon-only image
    b64 = _icon_b64()
    if b64:
        icon_html = f'<img src="data:image/png;base64,{b64}" alt="GreenGate icon" />'
    else:
        icon_html = '<div style="font-size:5rem;animation:iconFloat 5s ease-in-out infinite;">🌿</div>'

    st.markdown(f"""
    <div class="landing-wrap">
      <div class="landing-inner">
        <div class="landing-icon-col">{icon_html}</div>
        <div class="landing-text-col">
          <div class="landing-badge">🌱 SUSTAINABLE PORTFOLIO INTELLIGENCE</div>
    <div class="landing-title">Greengate</div>
          <div class="landing-subtitle">Your Door to Sustainable Investment Opportunities</div>
          <div class="landing-divider"></div>
          <div class="landing-tagline">Invest smarter. Invest greener.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Button centred under the layout
    _, btn_col, _ = st.columns([1, 1.1, 1])
    with btn_col:
        if st.button("Enter App  →", key="landing_enter_btn", use_container_width=True):
            st.session_state.entered = True
            st.rerun()

    render_sage("Greengate uses real market data and Refinitiv ESG scores to build a portfolio that reflects both your financial goals and your values. Takes about 3 minutes to set up.")



def render_sidebar_logo():
    """Place the icon-only logo at the top of the sidebar with transparent bg."""
    b64 = _icon_b64() or _logo_b64()
    if b64:
        st.sidebar.markdown(
            """<style>
            [data-testid="stSidebar"] img {
                background: transparent !important;
                mix-blend-mode: normal;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            f'<div style="text-align:center;padding:1rem 0 0.8rem 0;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:90px;width:72%;height:auto;'
            f'background:transparent !important;'
            f'filter:drop-shadow(0 4px 14px rgba(26,107,60,0.35));" '
            f'alt="GreenGate" />'
            f'</div>',
            unsafe_allow_html=True,
        )

def inject_apple_theme():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
          :root {
            --bg: #f0f7f2;
            --surface: #ffffff;
            --surface-soft: #f5faf6;
            --surface-muted: #e8f0ea;
            --border: #c5d9cc;
            --text: #1a2e1f;
            --muted: #5a7a63;
            --accent: #1a6b3c;
            --accent-soft: #e0f0e6;
            --green: #2d8a4e;
            --green-bright: #4caf7d;
            --sidebar-bg: #eef8f1;
            --sidebar-text: #254b33;
            --shadow: 0 10px 28px rgba(13, 51, 32, 0.08);
            --radius: 16px;
          }

          /* ---- Smooth fade-in animations ---- */
          @keyframes fadeInUp {
            0%   { opacity:0; transform:translateY(16px); }
            100% { opacity:1; transform:translateY(0); }
          }
          @keyframes fadeIn {
            0%   { opacity:0; }
            100% { opacity:1; }
          }
          @keyframes slideInLeft {
            0%   { opacity:0; transform:translateX(-14px); }
            100% { opacity:1; transform:translateX(0); }
          }

          /* ---- Subtle floating leaves: styles injected below ---- */

          /* Apply fade-in to main content blocks */
          .block-container > div > div > div {
            animation: fadeInUp 0.45s ease both;
          }
          [data-testid="stTabs"] [data-testid="stTabContent"] > div {
            animation: fadeIn 0.35s ease both;
          }
          [data-testid="stSidebar"] > div:first-child {
            animation: slideInLeft 0.4s ease both;
          }

          /* Smooth transitions on interactive elements */
          .stButton > button,
          .stDownloadButton > button,
          [data-testid="stExpander"],
          .metric-card,
          .green-card,
          .hero-card {
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
          }
          .stButton > button:hover,
          .stDownloadButton > button:hover {
            transform: translateY(-2px) scale(1.02);
          }
          .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 36px rgba(13, 51, 32, 0.14);
            border-color: var(--green-bright) !important;
          }
          .green-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(45, 138, 78, 0.15);
          }

          .stApp, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 10% 5%, #ffffff 0%, transparent 30%),
                        linear-gradient(180deg, #f5faf6 0%, var(--bg) 100%) !important;
            color: var(--text) !important;
            font-family: 'Inter', sans-serif;
          }

          [data-testid="stHeader"] {
            background: rgba(240, 247, 242, 0.94) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border);
          }

          /* ---- Sidebar: soft green matching the tab palette ---- */
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eef8f1 0%, #e5f2e9 100%) !important;
            border-right: 1px solid #c5d9cc !important;
          }

          [data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
          [data-testid="stSidebar"] .stMarkdown p,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] .stCaption { color: #426450 !important; }
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] h4 { color: #1f3b2a !important; }

          /* Sidebar inputs on light bg */
          [data-testid="stSidebar"] div[data-baseweb="input"] > div,
          [data-testid="stSidebar"] div[data-baseweb="select"] > div,
          [data-testid="stSidebar"] .stNumberInput > div > div,
          [data-testid="stSidebar"] .stTextInput > div > div,
          [data-testid="stSidebar"] textarea {
            background: #ffffff !important;
            border: 1px solid #c7d9cd !important;
            color: var(--sidebar-text) !important;
          }
          [data-testid="stSidebar"] div[data-baseweb="input"] input,
          [data-testid="stSidebar"] div[data-baseweb="select"] input,
          [data-testid="stSidebar"] .stTextInput input,
          [data-testid="stSidebar"] .stNumberInput input {
            color: #1f3b2a !important;
            -webkit-text-fill-color: #1f3b2a !important;
          }
          [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #1a6b3c 0%, #2d8a4e 100%) !important;
            border: 1px solid #2d8a4e !important;
          }
          .currency-prefix {
            display:flex;
            align-items:center;
            justify-content:center;
            height:2.5rem;
            margin-top:0.15rem;
            border:1px solid #c7d9cd;
            border-right:none;
            border-radius:10px 0 0 10px;
            background:rgba(255,255,255,0.82);
            color:#1f3b2a;
            font-weight:700;
          }

          /* ---- Main area text colours ---- */
          h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--text) !important; }
          .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

          /* ---- Tabs: green accent ---- */
          [data-testid="stTabs"] [role="tablist"] {
            background: var(--surface-soft) !important;
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border);
          }
          [data-testid="stTabs"] [role="tab"] { color: var(--muted) !important; border-radius: 9px; font-weight: 500; }
          [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #e8f5ee, #f0faf4) !important;
            color: var(--accent) !important;
            font-weight: 700;
            border-bottom: 3px solid var(--accent) !important;
            box-shadow: 0 2px 10px rgba(26,107,60,0.10);
          }
          [data-testid="stTabContent"] { background: transparent !important; }

          /* ---- Inputs ---- */
          div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, .stNumberInput > div > div, .stTextInput > div > div, textarea {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
          }
          div[data-baseweb="input"] input, div[data-baseweb="select"] input, .stTextInput input, .stNumberInput input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            background: transparent !important;
          }
          div[data-baseweb="input"] input::placeholder,
          .stTextInput input::placeholder,
          .stNumberInput input::placeholder {
            color: rgba(90, 122, 99, 0.45) !important;
            -webkit-text-fill-color: rgba(90, 122, 99, 0.45) !important;
            opacity: 1 !important;
          }
          .stNumberInput button,
          .stNumberInput [data-baseweb="input"] button {
            background: #f0f7f2 !important;
            color: #1a2e1f !important;
            border-left: 1px solid var(--border) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
          }
          .stNumberInput button:hover,
          .stNumberInput [data-baseweb="input"] button:hover {
            background: #e0f0e6 !important;
            color: #1a2e1f !important;
          }

          .stRadio label {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 10px;
            padding: 0.5rem 0.8rem;
            transition: border-color 0.2s ease, background 0.2s ease;
          }
          .stRadio label:hover { border-color: var(--green) !important; background: var(--surface-soft) !important; }
          [data-testid="stSlider"] * { color: var(--text) !important; }
          [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: var(--accent) !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 1px 4px rgba(13, 51, 32, 0.25) !important;
          }
          /* Slider rail + filled track */
          [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
            background: #c5d9cc !important;
          }
          [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div > div {
            background: linear-gradient(90deg, #1a6b3c, #2d8a4e) !important;
          }
          /* Sidebar slider overrides */
          [data-testid="stSidebar"] [data-testid="stSlider"] * { color: var(--sidebar-text) !important; }
          [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
            background: rgba(200,230,212,0.20) !important;
          }
          [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div > div {
            background: linear-gradient(90deg, #1a6b3c, #4caf7d) !important;
          }

          /* ---- Buttons: green gradient ---- */
          .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #1a6b3c 0%, #2d8a4e 100%) !important;
            color: #ffffff !important;
            border: 1px solid #1a6b3c !important;
            border-radius: 999px;
            padding: 0.68rem 1.35rem;
            font-weight: 600;
            box-shadow: 0 6px 18px rgba(26, 107, 60, 0.22);
          }
          .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #155a32 0%, #1a6b3c 100%) !important;
            border-color: #155a32 !important;
            box-shadow: 0 8px 24px rgba(26, 107, 60, 0.30);
          }
          .stButton > button p, .stButton > button span, .stDownloadButton > button p, .stDownloadButton > button span, [data-testid="stSidebar"] .stButton > button {
            color: #ffffff !important;
          }

          /* ---- Expanders: green accent ---- */
          [data-testid="stExpander"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-left: 3px solid var(--green-bright) !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 12px rgba(13, 51, 32, 0.05);
          }
          [data-testid="stExpander"] details,
          [data-testid="stExpander"] summary {
            background: #f5faf6 !important;
          }
          [data-testid="stExpander"] summary {
            color: #1a2e1f !important;
            font-weight: 600;
            border-bottom: 1px solid #d4e5da !important;
          }
          [data-testid="stExpander"] summary svg {
            fill: var(--accent) !important;
            color: var(--accent) !important;
          }
          [data-testid="stExpander"] * { color: var(--text) !important; }

          /* ---- Metric ---- */
          [data-testid="stMetric"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: var(--shadow);
          }
          [data-testid="stMetricLabel"] { color: var(--muted) !important; }
          [data-testid="stMetricValue"] { color: var(--accent) !important; }

          /* ---- Hero card ---- */
          .hero-card {
            background: linear-gradient(180deg, #ffffff, #f5faf6);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem 2.2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.4rem;
          }
          .hero-eyebrow {
            display: inline-block;
            padding: 0.34rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-bottom: 0.9rem;
          }
          .hero-subtitle { margin-top: 0.55rem; font-size: 1.1rem; font-weight: 600; color: #1a6b3c !important; }
          .hero-description { margin-top: 0.45rem; font-size: 0.95rem; line-height: 1.7; color: var(--muted) !important; max-width: 680px; }

          /* ---- Cards: white bg, green left border ---- */
          .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--green-bright);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-height: 130px;
          }
          .metric-card-label { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem; }
          .metric-card-value { font-size: clamp(1.2rem, 1.8vw, 2rem); font-weight: 700; color: var(--accent); letter-spacing: -0.03em; }

          .green-card {
            background: #f0f7f2;
            border: 1px solid #b8d4c2;
            border-left: 3px solid var(--green-bright);
            border-radius: 14px;
            padding: 1rem 1.2rem;
          }
          .green-card-title { font-size: 0.9rem; font-weight: 700; color: #1a6b3c; margin-bottom: 0.45rem; }
          .green-card-line { font-size: 0.95rem; color: #2a5a3e; line-height: 1.6; }

          .section-title { font-size: 1.25rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; margin: 1.2rem 0 0.7rem 0; }
          .warning-text { color: #b14444; font-size: 0.9rem; font-weight: 700; }
          .sidebar-profile { margin: 0.04rem 0 0.45rem 0; }
          .sidebar-company-name { font-size: 0.98rem; font-weight: 700; color: var(--sidebar-text); margin: 0 0 0.08rem 0; }
          .sidebar-company-name.sin-stock { color: #f87171; }
          .sidebar-company-line { font-size: 0.92rem; color: #426450; margin: 0; }

          [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow);
          }
          [data-testid="stDataFrame"] *, [data-testid="stTable"] * { color: var(--text) !important; background: transparent !important; }

          .stAlert {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-left: 3px solid var(--green-bright) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
          }

          div[data-baseweb="popover"] *, ul[role="listbox"], li[role="option"] { background: #ffffff !important; color: var(--text) !important; }
          li[role="option"]:hover { background: var(--surface-soft) !important; }

          ::-webkit-scrollbar { width: 8px; }
          ::-webkit-scrollbar-track { background: #e0ede4; }
          ::-webkit-scrollbar-thumb { background: #a8c5b2; border-radius: 6px; }
          ::-webkit-scrollbar-thumb:hover { background: #8bb19a; }

          .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

          /* Polished divider replacement */
          hr { border: none; border-top: 1px solid var(--border); margin: 1.6rem 0; }

          /* Progress bar: green gradient fill */
          .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #1a6b3c, #4caf7d) !important;
            transition: width 0.6s cubic-bezier(.16,1,.3,1);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inject subtle floating leaves for the main app background (richer set)
    st.markdown(
        """
        <style>
          @keyframes leafDrift {
            0%   { transform: translate(0, 0) rotate(0deg);   opacity: 0.10; }
            25%  { transform: translate(12px, -20px) rotate(8deg);  opacity: 0.16; }
            50%  { transform: translate(-8px, -38px) rotate(-5deg); opacity: 0.14; }
            75%  { transform: translate(16px, -18px) rotate(10deg); opacity: 0.18; }
            100% { transform: translate(0, 0) rotate(0deg);   opacity: 0.10; }
          }
          .app-leaf {
            position: fixed;
            pointer-events: none;
            z-index: 0;
            animation: leafDrift ease-in-out infinite;
            color: #2d8a4e;
            user-select: none;
          }
        </style>
        <!-- Bottom-right cluster -->
        <span class="app-leaf" style="bottom:8%;right:2%;font-size:2rem;animation-duration:18s;">🍃</span>
        <span class="app-leaf" style="bottom:18%;right:5%;font-size:1.4rem;animation-duration:23s;animation-delay:3s;">🍃</span>
        <span class="app-leaf" style="bottom:30%;right:1%;font-size:1.1rem;animation-duration:19s;animation-delay:7s;">☘️</span>
        <!-- Top-left cluster -->
        <span class="app-leaf" style="top:8%;left:1%;font-size:1.7rem;animation-duration:21s;animation-delay:5s;">🌿</span>
        <span class="app-leaf" style="top:20%;left:3%;font-size:1.2rem;animation-duration:25s;animation-delay:9s;">🍃</span>
        <!-- Scattered -->
        <span class="app-leaf" style="top:50%;left:0.5%;font-size:1.0rem;animation-duration:20s;animation-delay:2s;">☘️</span>
        <span class="app-leaf" style="bottom:45%;right:2%;font-size:1.3rem;animation-duration:22s;animation-delay:11s;">🍃</span>
        <span class="app-leaf" style="top:70%;left:2%;font-size:0.9rem;animation-duration:17s;animation-delay:6s;">🌿</span>
        """,
        unsafe_allow_html=True,
    )

def render_hero():
    # Use the icon-only image for the hero card, fall back to full logo
    b64 = _icon_b64() or _logo_b64()
    if b64:
        logo_src = f"data:image/png;base64,{b64}"
    else:
        logo_src = ""
    logo_html = ""
    if logo_src:
        logo_html = (
            f'<div style="display:flex;justify-content:flex-end;align-items:center;height:100%;">'
            f'<img src="{logo_src}" '
            f'style="max-width:220px;width:100%;height:auto;object-fit:contain;'
            f'filter:drop-shadow(0 12px 30px rgba(26,107,60,0.25));'
            f'animation:fadeIn 0.8s ease both 0.2s;" '
            f'alt="GreenGate icon" />'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="hero-card" style="animation:fadeInUp 0.5s ease both;">
          <div style="display:flex;gap:1.6rem;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;">
            <div style="flex:1 1 520px;min-width:320px;">
              <div class="hero-eyebrow">Sustainable Portfolio Intelligence</div>
              <h1 class="hero-title">
                <span style="color:#1a6b3c;font-size:clamp(2.8rem,5vw,4.2rem);font-weight:800;letter-spacing:-0.05em;">Green</span>
                <span style="color:#2d8a4e;font-size:clamp(2.8rem,5vw,4.2rem);font-weight:800;letter-spacing:-0.05em;">gate</span>
              </h1>
              <p class="hero-subtitle">Invest smarter. Invest greener.</p>
              <p class="hero-description">
                Enter your stocks and hit <strong style="color:#1a6b3c;">Run portfolio optimisation</strong>.
              </p>
            </div>
            <div style="flex:0 0 30%;max-width:280px;min-width:160px;align-self:stretch;">
              {logo_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def render_metric_card(column, label, value):
    column.markdown(f'<div class="metric-card"> <div class="metric-card-label">{label}</div> <div class="metric-card-value">{value}</div> </div>', unsafe_allow_html=True)

def render_green_cost_card(sharpe_gap, esg_gain):
    st.markdown(f'<div class="green-card"> <div class="green-card-title">Green Cost Calculator</div> <div class="green-card-line"><strong>Sharpe ratio difference:</strong> {sharpe_gap:+.3f}</div> <div class="green-card-line"><strong>ESG score gain:</strong> {esg_gain:+.2f} points</div> <div class="green-card-line"><strong>Formula:</strong> Tangency Sharpe ratio - Recommended Sharpe ratio</div> </div>', unsafe_allow_html=True)

def render_sin_warning():
    st.sidebar.markdown('<div class="warning-text">This is not green.</div>', unsafe_allow_html=True)

def render_sidebar_company_profile(profile, selected_sin_industries, esg_details=None):
    if not profile:
        if esg_details:
            esg_html = f"""<div class="sidebar-company-line">ESG: {esg_details['ESG']:.3f}</div><div class="sidebar-company-line">E: {esg_details['E']:.3f} | S: {esg_details['S']:.3f} | G: {esg_details['G']:.3f}</div>"""
        else:
            esg_html = '<div class="sidebar-company-line">ESG: Unavailable</div><div class="sidebar-company-line">E: Unavailable | S: Unavailable | G: Unavailable</div>'
        st.sidebar.markdown(f'<div class="sidebar-profile"><div class="sidebar-company-line">Sector: Unavailable</div><div class="sidebar-company-line">Industry: Unavailable</div>{esg_html}</div>', unsafe_allow_html=True)
        return
    is_sin = profile['industry'] in selected_sin_industries
    company_class = 'sidebar-company-name sin-stock' if is_sin else 'sidebar-company-name'
    warning_html = '<div class="warning-text">This is not green.</div>' if is_sin else ''
    if esg_details:
        esg_html = f"""<div class="sidebar-company-line">ESG: {esg_details['ESG']:.3f}</div><div class="sidebar-company-line">E: {esg_details['E']:.3f} | S: {esg_details['S']:.3f} | G: {esg_details['G']:.3f}</div>"""
    else:
        esg_html = '<div class="sidebar-company-line">ESG: Unavailable</div><div class="sidebar-company-line">E: Unavailable | S: Unavailable | G: Unavailable</div>'
    st.sidebar.markdown(f'''<div class="sidebar-profile"> {warning_html} <div class="{company_class}">{profile['name']}</div> <div class="sidebar-company-line">Sector: {profile['sector']}</div> <div class="sidebar-company-line">Industry: {profile['industry']}</div> {esg_html} </div>''', unsafe_allow_html=True)

def style_table(df):
    df = df.reset_index(drop=True)
    return (
        df.style.hide(axis='index')
        .set_table_styles(
            [
                {
                    'selector': 'table',
                    'props': [
                        ('width', 'max-content'),
                        ('border-collapse', 'separate'),
                        ('border-spacing', '0'),
                        ('background', '#ffffff'),
                        ('color', '#1f2933'),
                        ('border', '1px solid #d7dee5'),
                        ('border-radius', '16px'),
                        ('overflow', 'hidden'),
                        ('box-shadow', '0 8px 18px rgba(15, 23, 42, 0.06)'),
                        ('font-size', '0.92rem'),
                    ],
                },
                {
                    'selector': 'thead th',
                    'props': [
                        ('background', '#f8fbfe'),
                        ('color', '#1f2f46'),
                        ('font-weight', '700'),
                        ('text-align', 'left'),
                        ('padding', '7px 9px'),
                        ('border-bottom', '1px solid #dbe3ea'),
                        ('letter-spacing', '0.01em'),
                        ('white-space', 'nowrap'),
                    ],
                },
                {
                    'selector': 'tbody td',
                    'props': [
                        ('background', '#ffffff'),
                        ('color', '#1f2933'),
                        ('padding', '6px 9px'),
                        ('border-bottom', '1px solid #edf1f5'),
                        ('vertical-align', 'middle'),
                        ('white-space', 'nowrap'),
                    ],
                },
                {
                    'selector': 'tbody tr:nth-child(even) td',
                    'props': [
                        ('background', '#fbfdff'),
                    ],
                },
                {
                    'selector': 'tbody tr:hover td',
                    'props': [
                        ('background', '#f3f8fc'),
                    ],
                },
                {
                    'selector': 'tbody tr:last-child td',
                    'props': [
                        ('border-bottom', 'none'),
                    ],
                },
            ]
        )
    )

def _parse_percent(cell_value):
    if cell_value is None:
        return None
    text = str(cell_value).strip()
    if not text.endswith('%'):
        return None
    try:
        return float(text[:-1].replace(',', '').strip())
    except Exception:
        return None

def _is_w1_w2_weight_column(col_name):
    name = str(col_name).strip().lower()
    if 'weight' not in name:
        return False
    if 'risk-free' in name or 'risk free' in name:
        return False
    return True

def _weight_bar_html(percent_value):
    fill = max(0.0, min(100.0, float(percent_value)))
    return (
        '<div style="display:flex;align-items:center;gap:0.45rem;">'
        '<div style="width:78px;height:10px;border-radius:7px;background:#d9e3f1;overflow:hidden;border:1px solid #cfdae6;">'
        f'<div style="width:{fill:.2f}%;height:100%;background:#7f9bc2;"></div>'
        '</div>'
        f'<span>{percent_value:.2f}%</span>'
        '</div>'
    )

def render_table(df):
    view = df.copy()
    asset_col_name = None
    for col in view.columns:
        if str(col).strip().lower() == 'asset':
            asset_col_name = col
            break

    for col in view.columns:
        if not _is_w1_w2_weight_column(col):
            continue
        for idx in view.index:
            value = _parse_percent(view.at[idx, col])
            if value is None:
                continue
            if asset_col_name is not None:
                asset_label = str(view.at[idx, asset_col_name]).strip().lower()
                if 'risk-free' in asset_label or 'risk free' in asset_label:
                    continue
            view.at[idx, col] = _weight_bar_html(value)

    st.markdown(style_table(view).to_html(escape=False), unsafe_allow_html=True)

def render_complete_portfolio_comparison(tangency, complete_portfolio, tickers):
    def pct(v):
        return f"{v * 100:.2f}%"

    def pct_diff(complete, base):
        return f"{(complete - base) * 100:+.2f}%"

    def num_diff(complete, base, digits=4):
        return f"{(complete - base):+.{digits}f}"

    def weight_bar(value, tone):
        fill = max(0, min(100, value * 100))
        if tone == 'tan':
            fill_color = '#7f9bc2'
            bg_color = '#d9e3f1'
        else:
            fill_color = '#6b847a'
            bg_color = '#d8e3de'
        return (
            f'<div style="display:flex;align-items:center;gap:0.5rem;">'
            f'<div style="width:84px;height:12px;border-radius:8px;background:{bg_color};overflow:hidden;border:1px solid #cfdae6;">'
            f'<div style="width:{fill:.2f}%;height:100%;background:{fill_color};"></div>'
            f'</div>'
            f'<span>{pct(value)}</span>'
            f'</div>'
        )

    ret_delta = complete_portfolio['Expected_Return'] - tangency['Expected_Return']
    vol_delta = complete_portfolio['Risk_SD'] - tangency['Risk_SD']
    util_val = complete_portfolio['Utility']
    tan_weights = tangency.get('Weights', {})
    comp_weights = complete_portfolio.get('Weights', {})
    rows = []
    for ticker in tickers:
        tan_weight = float(tan_weights.get(ticker, 0.0))
        comp_weight = float(comp_weights.get(ticker, 0.0))
        rows.append({
            'metric': f'Weight in {ticker}',
            'tan': weight_bar(tan_weight, 'tan'),
            'comp': weight_bar(comp_weight, 'comp'),
            'diff': pct_diff(comp_weight, tan_weight),
        })
    rows.extend([
        {
            'metric': 'Weight in risk-free asset',
            'tan': weight_bar(0.0, 'tan'),
            'comp': weight_bar(complete_portfolio['weight_risk_free'], 'comp'),
            'diff': pct_diff(complete_portfolio['weight_risk_free'], 0.0),
        },
        {
            'metric': 'Expected return',
            'tan': pct(tangency['Expected_Return']),
            'comp': pct(complete_portfolio['Expected_Return']),
            'diff': f'{abs(ret_delta) * 100:.2f}% lower' if ret_delta < 0 else f'{ret_delta * 100:.2f}% higher',
        },
        {
            'metric': 'Volatility',
            'tan': pct(tangency['Risk_SD']),
            'comp': pct(complete_portfolio['Risk_SD']),
            'diff': f'{abs(vol_delta) * 100:.2f}% lower risk' if vol_delta < 0 else f'{vol_delta * 100:.2f}% higher risk',
        },
        {
            'metric': 'Variance',
            'tan': f"{tangency['Variance']:.4f}",
            'comp': f"{complete_portfolio['Variance']:.4f}",
            'diff': num_diff(complete_portfolio['Variance'], tangency['Variance']),
        },
        {
            'metric': 'Sharpe ratio',
            'tan': f"{tangency['Sharpe_Ratio']:.3f}",
            'comp': f"{tangency['Sharpe_Ratio']:.3f}",
            'diff': 'Same Sharpe',
        },
        {
            'metric': 'Utility',
            'tan': 'N/A',
            'comp': f"{util_val:.4f}",
            'diff': f"{util_val:.4f} higher utility",
        },
    ])

    row_html = ''.join(
        f"""
        <tr>
          <td style="padding:0.56rem 0.78rem;font-weight:600;color:#1f2933;font-size:0.96rem;line-height:1.3;white-space:nowrap;">{r['metric']}</td>
          <td style="padding:0.56rem 0.78rem;color:#243b57;font-size:0.95rem;line-height:1.3;white-space:nowrap;">{r['tan']}</td>
          <td style="padding:0.56rem 0.78rem;color:#2f5a48;font-size:0.95rem;line-height:1.3;white-space:nowrap;">{r['comp']}</td>
          <td style="padding:0.56rem 0.78rem;color:#1f2933;font-size:0.92rem;line-height:1.25;white-space:nowrap;">
            <span style="display:inline-block;padding:0.22rem 0.65rem;border-radius:999px;background:#edf3ef;border:1px solid #d3e0d9;color:#2f5a48;font-weight:600;">
              {r['diff']}
            </span>
          </td>
        </tr>
        """
        for r in rows
    )

    st.markdown(
        f"""
        <div style="overflow-x:auto;margin-top:0.15rem;">
          <div style="display:inline-block;width:max-content;max-width:100%;background:#ffffff;border:1px solid #d7dee5;border-radius:16px;padding:0.5rem 0.6rem;box-shadow:0 8px 18px rgba(15,23,42,0.06);">
          <table style="width:max-content;min-width:1130px;border-collapse:separate;border-spacing:0;font-size:0.95rem;table-layout:auto;">
            <thead>
              <tr>
                <th style="text-align:left;padding:0.62rem 0.78rem;color:#1f2f46;border-bottom:1px solid #e7edf3;font-size:0.95rem;white-space:nowrap;">Metric</th>
                <th style="text-align:left;padding:0.62rem 0.78rem;color:#1f2f46;border-bottom:1px solid #e7edf3;font-size:0.95rem;white-space:nowrap;">Tangency Portfolio</th>
                <th style="text-align:left;padding:0.62rem 0.78rem;color:#1f2f46;border-bottom:1px solid #e7edf3;font-size:0.95rem;white-space:nowrap;">Complete Portfolio</th>
                <th style="text-align:left;padding:0.62rem 0.78rem;color:#6b7280;border-bottom:1px solid #e7edf3;font-size:0.95rem;white-space:nowrap;">Difference</th>
              </tr>
            </thead>
            <tbody>{row_html}</tbody>
          </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_recommendation_summary(tickers, recommended, tangency, complete_portfolio, market_data, compatibility_by_asset, gamma, lambda_raw_avg, w_e, w_s, w_g, investment_amount, risk_free_rate, profiles):
    sharpe_gap = tangency['Sharpe_Ratio'] - recommended['Sharpe_Ratio']
    esg_gain = recommended['ESG_Score'] - tangency['ESG_Score']
    composites = [item.get('composite') for item in compatibility_by_asset if item.get('composite') is not None]
    esg_ok = any(score >= 45 for score in composites)
    green_cost_ok = abs(sharpe_gap) <= 0.15
    return_ok = recommended['Expected_Return'] > risk_free_rate + 0.02
    if esg_ok and green_cost_ok and return_ok:
        verdict, verdict_color, verdict_bg, verdict_border = ('Proceed with Confidence', '#166534', 'rgba(52,199,89,0.1)', 'rgba(52,199,89,0.3)')
    elif esg_ok or (green_cost_ok and return_ok):
        verdict, verdict_color, verdict_bg, verdict_border = ('Proceed with Awareness', '#92400e', 'rgba(245,158,11,0.1)', 'rgba(245,158,11,0.3)')
    else:
        verdict, verdict_color, verdict_bg, verdict_border = ('Consider Alternatives', '#991b1b', 'rgba(239,68,68,0.1)', 'rgba(239,68,68,0.3)')
    risk_label = classify_risk(gamma)
    esg_label = classify_esg(lambda_raw_avg)
    dim_weights = {'E': w_e, 'S': w_s, 'G': w_g}
    primary_dim = max(dim_weights, key=dim_weights.get)
    dim_full = {'E': 'Environmental', 'S': 'Social', 'G': 'Governance'}
    primary_focus = dim_full[primary_dim]
    rec_ret = recommended['Expected_Return'] * 100
    rec_vol = recommended['Risk_SD'] * 100
    rec_sharpe = recommended['Sharpe_Ratio']
    rec_esg = recommended['ESG_Score']
    tan_sharpe = tangency['Sharpe_Ratio']
    if abs(sharpe_gap) < 0.02:
        green_cost_text = 'at virtually no cost to risk-adjusted performance'
    elif sharpe_gap > 0:
        green_cost_text = f'accepting a modest {abs(sharpe_gap):.3f} Sharpe ratio reduction for a {esg_gain:+.2f} point ESG gain'
    else:
        green_cost_text = f'while actually improving the Sharpe ratio by {abs(sharpe_gap):.3f}'
    rec_weights = recommended.get('Weights', {})
    ordered_allocations = [(ticker, float(rec_weights.get(ticker, 0.0))) for ticker in tickers]
    allocation_bits = []
    amount_bits = []
    for ticker, weight in ordered_allocations:
        sector = (profiles.get(ticker) or {}).get('sector') or ''
        sector_text = f' ({sector})' if sector else ''
        allocation_bits.append(f'<strong>{weight * 100:.1f}%</strong> to <strong>{ticker}</strong>{sector_text}')
        if investment_amount > 0:
            amount_bits.append(f'<strong>${investment_amount * weight:,.2f}</strong> in {ticker}')
    allocation_text = ', '.join(allocation_bits[:-1]) + (f' and {allocation_bits[-1]}' if len(allocation_bits) > 1 else allocation_bits[0] if allocation_bits else '')
    amount_str = ''
    if investment_amount > 0 and amount_bits:
        amount_text = ', '.join(amount_bits[:-1]) + (f' and {amount_bits[-1]}' if len(amount_bits) > 1 else amount_bits[0])
        amount_str = f' Investing <strong>${investment_amount:,.0f}</strong> here means {amount_text}.'
    points = []
    if rec_sharpe > 0.5:
        points.append(f'Strong risk-adjusted return  Sharpe of {rec_sharpe:.3f} indicates attractive reward per unit of risk.')
    else:
        points.append(f'Moderate risk-adjusted return  consider whether {rec_ret:.2f}% expected return justifies {rec_vol:.2f}% volatility.')
    if abs(sharpe_gap) < 0.05:
        points.append('Minimal green cost  your ESG preferences are financially compatible with this portfolio.')
    elif sharpe_gap > 0.1:
        points.append(f'Green cost of {sharpe_gap:.3f} SR  this reflects a conscious values-vs-returns trade-off.')
    for item in compatibility_by_asset:
        ticker = item['ticker']
        composite = item.get('composite')
        if composite is not None and composite >= 70:
            points.append(f'{ticker} is a high-compatibility holding  its ESG profile closely mirrors your stated values.')
        elif composite is not None and composite < 45:
            points.append(f'Review {ticker} (compatibility {composite:.1f}/100)  check the Alternative Recommendations for better-aligned options.')
    avg_corr = market_data.get('avg_correlation')
    if avg_corr is not None and avg_corr < 0.3:
        points.append(f"Low average asset correlation ({avg_corr:.2f})  strong diversification benefit in this portfolio.")
    elif avg_corr is not None and avg_corr > 0.7:
        points.append(f"High average asset correlation ({avg_corr:.2f})  consider adding a less correlated asset for stronger diversification.")
    compatibility_bits = []
    for item in compatibility_by_asset:
        composite = item.get('composite')
        align_label = 'strong alignment' if composite and composite >= 70 else 'moderate alignment' if composite and composite >= 45 else 'weak alignment'
        detail = f'(score: {composite:.1f}/100)' if composite is not None else '(data unavailable)'
        compatibility_bits.append(f'<strong>{item["ticker"]}</strong> shows <strong>{align_label}</strong> {detail}')
    compatibility_text = ' '.join(compatibility_bits)
    points_html = ''.join((f'<div style="display:flex;gap:0.6rem;align-items:flex-start;margin-bottom:0.4rem;"><span style="color:#2d5c88;font-weight:700;flex-shrink:0;"></span><span style="font-size:0.92rem;line-height:1.6;color:#4f5f6e;">{p}</span></div>' for p in points[:6]))
    st.markdown(f"""<div style="background:#ffffff; border:1px solid #d7dee5;border-radius:20px;padding:2rem 2.2rem; box-shadow:0 10px 28px rgba(15,23,42,0.08);margin-top:0.5rem;"> <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.6rem;flex-wrap: wrap;"> <div style="background:{verdict_bg};border:1.5px solid {verdict_border};border-radius:12px; padding:0.55rem 1.2rem;"> <span style="font-size:1rem;font-weight:700;color:{verdict_color};">{verdict}</span> </div> <div style="font-size:0.82rem;color:#667380;font-weight:500;">Portfolio recommendation summary</div> </div> <div style="margin-bottom:1.4rem;"> <div style="font-size:0.74rem;font-weight:700;color:#2d5c88;letter-spacing:0.07em;margin-bottom:0.5rem;">RECOMMENDED ALLOCATION</div> <p style="font-size:0.96rem;line-height:1.8;color:#1f2933;margin:0;"> As a <strong>{risk_label}</strong>, <strong>{esg_label}</strong> investor with a primary focus on <strong>{primary_focus}</strong>, your optimal portfolio allocates {allocation_text}. Expected annual return: <strong>{rec_ret:.2f}%</strong> at <strong>{rec_vol:.2f}%</strong> annualised volatility (Sharpe ratio: <strong>{rec_sharpe:.3f}</strong>).{amount_str} </p> </div> <div style="margin-bottom:1.4rem;"> <div style="font-size:0.74rem;font-weight:700;color:#3e5d78;letter-spacing:0.07em;margin-bottom:0.5rem;">RISK &amp; SUSTAINABILITY BALANCE</div> <p style="font-size:0.96rem;line-height:1.8;color:#1f2933;margin:0;"> Your ESG-adjusted portfolio achieves a combined ESG score of <strong>{rec_esg:.2f}</strong>, {green_cost_text}. The pure financial optimum (tangency portfolio) offers a Sharpe ratio of <strong>{tan_sharpe:.3f}</strong> vs. <strong>{rec_sharpe:.3f}</strong> for your ESG-weighted portfolio  reflecting your sustainability commitment of <strong>{lambda_raw_avg:.1f}/4</strong>. </p> </div> <div style="margin-bottom:1.4rem;"> <div style="font-size:0.74rem;font-weight:700;color:#2f7a59;letter-spacing:0.07em;margin-bottom:0.5rem;">ESG COMPATIBILITY</div> <p style="font-size:0.96rem;line-height:1.8;color:#1f2933;margin:0;"> {compatibility_text} Your {primary_focus} emphasis ({dim_weights[primary_dim]:.0%} weight) is the primary lens through which these holdings are evaluated. </p> </div> <div> <div style="font-size:0.74rem;font-weight:700;color:#667380;letter-spacing:0.07em;margin-bottom:0.65rem;">KEY TAKEAWAYS</div> {points_html} </div> </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FEATURE 1 — Animated ESG Score Gauge (matplotlib semicircular arc)
# ---------------------------------------------------------------------------

def render_esg_gauge(score):
    """Draw a semicircular arc gauge for the portfolio ESG score (0-100).

    The gauge fills proportionally to *score*, coloured on a red-amber-green
    gradient.  The numeric score is shown in the centre, with a label below.
    Rendered via matplotlib and displayed centred using st.columns.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    score = float(max(0.0, min(100.0, score)))

    # Pick colour on a red -> amber -> green gradient based on score
    if score < 50:
        t = score / 50.0
        # interpolate #ef4444 -> #f59e0b
        r = int(0xef + t * (0xf5 - 0xef))
        g = int(0x44 + t * (0x9e - 0x44))
        b = int(0x44 + t * (0x0b - 0x44))
    else:
        t = (score - 50) / 50.0
        # interpolate #f59e0b -> #34c759
        r = int(0xf5 + t * (0x34 - 0xf5))
        g = int(0x9e + t * (0xc7 - 0x9e))
        b = int(0x0b + t * (0x59 - 0x0b))
    arc_color = f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_alpha(0.0)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.25)
    ax.axis("off")

    # Background arc (light grey, full semicircle 0-180 degrees)
    bg_arc = mpatches.Arc(
        (0, 0), 2.0, 2.0, angle=0, theta1=0, theta2=180,
        linewidth=18, color="#e0e0e0", capstyle="round",
    )
    ax.add_patch(bg_arc)

    # Filled arc proportional to score
    fill_angle = (score / 100.0) * 180.0
    if fill_angle > 0.5:
        fill_arc = mpatches.Arc(
            (0, 0), 2.0, 2.0, angle=0, theta1=0, theta2=fill_angle,
            linewidth=18, color=arc_color, capstyle="round",
        )
        ax.add_patch(fill_arc)

    # Score text in centre
    ax.text(
        0, 0.38, f"{score:.0f}",
        ha="center", va="center",
        fontsize=36, fontweight="bold", color=arc_color,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        0, 0.08, "/ 100",
        ha="center", va="center",
        fontsize=11, color="#7b90b8",
        fontfamily="DejaVu Sans",
    )
    # Label below
    ax.text(
        0, -0.22, "Portfolio ESG Score",
        ha="center", va="center",
        fontsize=12, fontweight="semibold", color="#1a2e1f",
        fontfamily="DejaVu Sans",
    )

    plt.tight_layout(pad=0.1)

    # Centre the gauge using columns
    _, gauge_col, _ = st.columns([1, 1.2, 1])
    with gauge_col:
        st.pyplot(fig, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FEATURE 2 — Investor Profile Card in Sidebar
# ---------------------------------------------------------------------------

def render_sidebar_profile_card(gamma, lambda_raw_avg, w_e, w_s, w_g):
    """Render a compact investor-profile card in the Streamlit sidebar.

    Shows risk type (colour-coded), ESG intensity, gamma/lambda stats,
    and an E/S/G proportion bar.  Only renders when the investor profile
    has been completed; otherwise shows a muted placeholder message.
    All output uses st.sidebar.markdown (HTML) to avoid widget misalignment.
    """
    # Guard: only show the full card when the profile is complete
    if not st.session_state.get("profile_complete"):
        st.sidebar.markdown(
            '<div style="font-size:0.78rem;color:#9abfa8;padding:0.6rem 0;">'
            'Complete your investor profile to see your summary here.</div>',
            unsafe_allow_html=True,
        )
        return

    risk_label = classify_risk(gamma)
    esg_label = classify_esg(lambda_raw_avg)

    # Risk colour: Defensive = green, Balanced = amber, Growth-Oriented = red
    risk_color_map = {
        "Defensive": "#34c759",
        "Balanced": "#f59e0b",
        "Growth-Oriented": "#ef4444",
    }
    risk_color = risk_color_map.get(risk_label, "#c8e6d4")

    # ESG intensity colour (use labels from classify_esg)
    esg_color_map = {
        "Sustainability-Led": "#34c759",
        "ESG-Aware": "#74b99a",
        "Low ESG Priority": "#f59e0b",
    }
    esg_color = esg_color_map.get(esg_label, "#c8e6d4")

    # E/S/G bar widths (percentages of total)
    total_w = w_e + w_s + w_g
    if total_w > 0:
        pct_e = w_e / total_w * 100
        pct_s = w_s / total_w * 100
        pct_g = w_g / total_w * 100
    else:
        pct_e = pct_s = pct_g = 33.3

    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(26,107,60,0.25), rgba(13,51,32,0.40));
        border: 1px solid rgba(200,230,212,0.18);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0 0.6rem 0;
    ">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.07em;
                    color:#c8e6d4;margin-bottom:0.65rem;text-transform:uppercase;">
            Your Investor Profile
        </div>

        <!-- Risk type -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
            <span style="font-size:0.78rem;color:#9abfa8;">Risk Type</span>
            <span style="font-size:0.82rem;font-weight:700;color:{risk_color};">{risk_label}</span>
        </div>

        <!-- ESG intensity -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.55rem;">
            <span style="font-size:0.78rem;color:#9abfa8;">ESG Intensity</span>
            <span style="font-size:0.82rem;font-weight:700;color:{esg_color};">{esg_label}</span>
        </div>

        <!-- Gamma / Lambda side by side -->
        <div style="display:flex;gap:0.5rem;margin-bottom:0.6rem;">
            <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:8px;padding:0.35rem 0.5rem;text-align:center;">
                <div style="font-size:0.65rem;color:#9abfa8;">gamma</div>
                <div style="font-size:1rem;font-weight:700;color:#e0f0e6;">{gamma:.2f}</div>
            </div>
            <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:8px;padding:0.35rem 0.5rem;text-align:center;">
                <div style="font-size:0.65rem;color:#9abfa8;">lambda</div>
                <div style="font-size:1rem;font-weight:700;color:#e0f0e6;">{lambda_raw_avg:.2f}</div>
            </div>
        </div>

        <!-- E/S/G weights bar -->
        <div style="font-size:0.68rem;color:#9abfa8;margin-bottom:0.25rem;">E / S / G weights</div>
        <div style="display:flex;height:10px;border-radius:999px;overflow:hidden;background:#0a2819;">
            <div style="width:{pct_e:.1f}%;background:#2d8a4e;" title="E: {pct_e:.0f}%"></div>
            <div style="width:{pct_s:.1f}%;background:#74b99a;" title="S: {pct_s:.0f}%"></div>
            <div style="width:{pct_g:.1f}%;background:#c8e6d4;" title="G: {pct_g:.0f}%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.62rem;color:#9abfa8;margin-top:0.2rem;">
            <span>E {pct_e:.0f}%</span>
            <span>S {pct_s:.0f}%</span>
            <span>G {pct_g:.0f}%</span>
        </div>
    </div>
    """
    st.sidebar.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FEATURE 4 — First-visit Onboarding Overlay
# ---------------------------------------------------------------------------

def render_onboarding_overlay():
    """Show a first-visit onboarding card explaining the app in 3 steps.

    Called via the early-return pattern in app.py: if show_onboarding is True,
    this renders and then app.py calls st.stop() so no other content appears.
    The "Got it" button sets show_onboarding = False and reruns.
    """
    st.markdown(
        """
        <style>
        .onboarding-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            padding: 2rem 0;
        }
        .onboarding-card {
            background: #ffffff;
            border-radius: 24px;
            padding: 2.4rem 2.6rem;
            max-width: 520px;
            width: 100%;
            box-shadow: 0 24px 64px rgba(13, 51, 32, 0.18);
            animation: fadeInUp 0.45s ease both;
        }
        .onboarding-badge {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            background: #e8f5ee;
            color: #1a6b3c;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-bottom: 0.8rem;
        }
        .onboarding-title {
            font-size: 1.6rem;
            font-weight: 800;
            color: #1a5c2a;
            margin-bottom: 0.4rem;
            letter-spacing: -0.03em;
        }
        .onboarding-subtitle {
            font-size: 0.92rem;
            color: #5a7a63;
            margin-bottom: 1.2rem;
            line-height: 1.5;
        }
        .onboarding-step {
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
            margin-bottom: 0.75rem;
        }
        .onboarding-step-num {
            flex-shrink: 0;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #1a6b3c, #2d8a4e);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .onboarding-step-text {
            font-size: 0.92rem;
            color: #1a2e1f;
            line-height: 1.5;
            padding-top: 0.15rem;
        }
        .onboarding-pulse {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #34c759;
            margin-right: 0.5rem;
            animation: pulse-green 2s ease-in-out infinite;
            vertical-align: middle;
        }
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 0 0 rgba(52,199,89,0.5); }
            50%       { box-shadow: 0 0 0 10px rgba(52,199,89,0); }
        }
        </style>
        <div class="onboarding-wrap">
        <div class="onboarding-card">
            <div class="onboarding-badge"><span class="onboarding-pulse"></span>WELCOME TO GREENGATE</div>
            <div class="onboarding-title">Get started in 3 steps</div>
            <div class="onboarding-subtitle">Build a sustainable portfolio tailored to your values and risk profile.</div>
            <div class="onboarding-step">
                <div class="onboarding-step-num">1</div>
                <div class="onboarding-step-text"><strong>Complete your investor profile</strong> in the first tab to define your risk appetite and ESG preferences.</div>
            </div>
            <div class="onboarding-step">
                <div class="onboarding-step-num">2</div>
                <div class="onboarding-step-text"><strong>Enter stock tickers</strong> in the sidebar and click <em>Run portfolio optimisation</em>.</div>
            </div>
            <div class="onboarding-step">
                <div class="onboarding-step-num">3</div>
                <div class="onboarding-step-text"><strong>Review your personalised ESG portfolio</strong> across the Dashboard, Analysis, and Breakdown tabs.</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Got it  -->", key="onboarding_dismiss", width="stretch"):
        st.session_state["show_onboarding"] = False
        st.rerun()




