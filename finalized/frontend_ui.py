# Frontend UI helpers (Streamlit layout, theme, and rendering components)

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from backend_portfolio import classify_risk, classify_esg

def inject_apple_theme():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
          :root {
            --bg: #f4f6f8;
            --surface: #ffffff;
            --surface-soft: #f8fafb;
            --surface-muted: #eef2f5;
            --border: #d7dee5;
            --text: #1f2933;
            --muted: #667380;
            --accent: #2d5c88;
            --accent-soft: #e7eff6;
            --green: #2f7a59;
            --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
            --radius: 16px;
          }

          .stApp, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 12% 8%, #ffffff 0%, transparent 32%),
                        linear-gradient(180deg, #f7f9fb 0%, var(--bg) 100%) !important;
            color: var(--text) !important;
            font-family: 'Inter', sans-serif;
          }

          [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.94) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border);
          }

          [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid var(--border) !important;
          }

          [data-testid="stSidebar"] * { color: var(--text) !important; }
          [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: var(--muted) !important; }
          h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--text) !important; }
          .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

          [data-testid="stTabs"] [role="tablist"] {
            background: var(--surface-soft) !important;
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border);
          }
          [data-testid="stTabs"] [role="tab"] { color: var(--muted) !important; border-radius: 9px; font-weight: 500; }
          [data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--surface) !important; color: var(--accent) !important; font-weight: 600; }
          [data-testid="stTabContent"] { background: transparent !important; }

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
            color: rgba(102, 115, 128, 0.45) !important;
            -webkit-text-fill-color: rgba(102, 115, 128, 0.45) !important;
            opacity: 1 !important;
          }
          .stNumberInput button,
          .stNumberInput [data-baseweb="input"] button {
            background: #f7fafc !important;
            color: #2b3b49 !important;
            border-left: 1px solid #d7dee5 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
          }
          .stNumberInput button:hover,
          .stNumberInput [data-baseweb="input"] button:hover {
            background: #eef3f8 !important;
            color: #1f2933 !important;
          }

          .stRadio label {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 10px;
            padding: 0.5rem 0.8rem;
          }
          .stRadio label:hover { border-color: #b9c6d2 !important; background: var(--surface-soft) !important; }
          [data-testid="stSlider"] * { color: var(--text) !important; }
          [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: var(--accent) !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.22) !important;
          }
          /* Slider rail + filled track (BaseWeb structure) */
          [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
            background: #d7dee5 !important;
          }
          [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div > div {
            background: #2d5c88 !important;
          }

          .stButton > button, .stDownloadButton > button {
            background: #2d5c88 !important;
            color: #ffffff !important;
            border: 1px solid #274f76 !important;
            border-radius: 999px;
            padding: 0.68rem 1.35rem;
            font-weight: 600;
            box-shadow: 0 6px 14px rgba(45, 92, 136, 0.18);
          }
          .stButton > button:hover, .stDownloadButton > button:hover { background: #274f76 !important; border-color: #274f76 !important; }
          .stButton > button p, .stButton > button span, .stDownloadButton > button p, .stDownloadButton > button span, [data-testid="stSidebar"] .stButton > button {
            color: #ffffff !important;
          }

          [data-testid="stExpander"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
          }
          [data-testid="stExpander"] details,
          [data-testid="stExpander"] summary {
            background: #f5f8fb !important;
          }
          [data-testid="stExpander"] summary {
            color: #1f2933 !important;
            font-weight: 600;
            border-bottom: 1px solid #e5ebf1 !important;
          }
          [data-testid="stExpander"] summary svg {
            fill: #2d5c88 !important;
            color: #2d5c88 !important;
          }
          [data-testid="stExpander"] * { color: var(--text) !important; }

          [data-testid="stMetric"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: var(--shadow);
          }
          [data-testid="stMetricLabel"] { color: var(--muted) !important; }
          [data-testid="stMetricValue"] { color: var(--text) !important; }

          .hero-card {
            background: linear-gradient(180deg, #ffffff, #fbfcfd);
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
          .hero-subtitle { margin-top: 0.55rem; font-size: 1.1rem; font-weight: 600; color: #2b3b49 !important; }
          .hero-description { margin-top: 0.45rem; font-size: 0.95rem; line-height: 1.7; color: var(--muted) !important; max-width: 680px; }

          .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-height: 130px;
          }
          .metric-card-label { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem; }
          .metric-card-value { font-size: clamp(1.2rem, 1.8vw, 2rem); font-weight: 700; color: var(--text); letter-spacing: -0.03em; }

          .green-card {
            background: #f2f8f5;
            border: 1px solid #cadfd3;
            border-radius: 14px;
            padding: 1rem 1.2rem;
          }
          .green-card-title { font-size: 0.9rem; font-weight: 700; color: #2f7a59; margin-bottom: 0.45rem; }
          .green-card-line { font-size: 0.95rem; color: #355a49; line-height: 1.6; }

          .section-title { font-size: 1.25rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; margin: 1.2rem 0 0.7rem 0; }
          .warning-text { color: #b14444; font-size: 0.9rem; font-weight: 700; }
          .sidebar-profile { margin: 0.04rem 0 0.45rem 0; }
          .sidebar-company-name { font-size: 0.98rem; font-weight: 700; color: var(--text); margin: 0 0 0.08rem 0; }
          .sidebar-company-name.sin-stock { color: #b14444; }
          .sidebar-company-line { font-size: 0.92rem; color: var(--muted); margin: 0; }

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
            border-radius: 14px !important;
            color: var(--text) !important;
          }

          div[data-baseweb="popover"] *, ul[role="listbox"], li[role="option"] { background: #ffffff !important; color: var(--text) !important; }
          li[role="option"]:hover { background: var(--surface-soft) !important; }

          ::-webkit-scrollbar { width: 8px; }
          ::-webkit-scrollbar-track { background: #e8edf2; }
          ::-webkit-scrollbar-thumb { background: #c2ced8; border-radius: 6px; }
          ::-webkit-scrollbar-thumb:hover { background: #a7b7c4; }

          .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero():
    logo_path = Path(__file__).with_name("ESG logo.png")
    logo_src = "https://raw.githubusercontent.com/jyew0812/Sustainable-Finance/2c5bfb9bdef7e7c6aa2007c3783ef4170b30827b/finalized/ESG%20logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_src = f"data:image/png;base64,{logo_b64}"
    logo_html = (
        f'<div style="display:flex;justify-content:flex-end;align-items:center;height:100%;">'
        f'<img src="{logo_src}" '
        f'style="max-width:560px;width:100%;height:auto;object-fit:contain;filter:drop-shadow(0 12px 30px rgba(0,0,0,0.38));" '
        f'alt="Greengate logo" />'
        f'</div>'
    )

    st.markdown(
        f"""
        <div class="hero-card">
          <div style="display:flex;gap:1.6rem;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;">
            <div style="flex:1 1 520px;min-width:320px;">
              <div class="hero-eyebrow">Sustainable Portfolio Intelligence</div>
              <h1 class="hero-title">
                <span style="color:#2d5c88;font-size:clamp(2.8rem,5vw,4.2rem);font-weight:800;letter-spacing:-0.05em;">Green</span>
                <span style="color:#2f7a59;font-size:clamp(2.8rem,5vw,4.2rem);font-weight:800;letter-spacing:-0.05em;">gate</span>
              </h1>
              <p class="hero-subtitle">Invest smarter. Invest greener.</p>
              <p class="hero-description">
                Build a personalised multi-asset portfolio optimised for your risk profile and ESG values powered by
                mean-variance theory and real market data. Complete your investor profile in the first tab, enter your
                stocks in the sidebar, then hit <strong style="color:#2d5c88;">Run portfolio optimisation</strong>.
              </p>
            </div>
            <div style="flex:0 0 52%;max-width:620px;min-width:320px;align-self:stretch;">
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




