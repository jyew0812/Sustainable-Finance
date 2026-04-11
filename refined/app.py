import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import frontend_ui as ui

from backend_portfolio import (
    SIN_INDUSTRIES, ESG_DEFAULT_SOURCE, weighted_esg,
    _ticker_variants, load_esg_data_from_path,
    fetch_ticker_profile, fetch_market_data, fetch_universe_returns,
    portfolio_return, portfolio_variance,
    portfolio_esg, esg_utility_function, build_portfolio_table,
    select_recommended_portfolio, select_max_sharpe_portfolio,
    build_complete_portfolio, classify_risk, classify_esg,
    make_frontier_figure, make_cml_figure, make_esg_tradeoff_figure,
    make_esg_efficient_frontier_figure, make_price_history_figure,
    compute_portfolio_compatibility, get_weight_vector,
)
from frontend_ui import (
    inject_apple_theme, render_hero, render_section_title, render_metric_card,
    render_green_cost_card, render_sidebar_company_profile,
    render_table, render_recommendation_summary, style_table,
    render_landing_page, render_sidebar_logo, ensure_icon_cropped,
    render_esg_gauge, render_sidebar_profile_card, render_onboarding_overlay,
    render_sage,
)

st.set_page_config(page_title="Greengate", page_icon="🌿", layout="wide")

# Ensure the icon-only crop exists (runs once, then no-ops)
ensure_icon_cropped()

inject_apple_theme()


def _sync_sidebar_visibility(should_open: bool):
    action = "open" if should_open else "close"
    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        const openBtn =
            doc.querySelector('button[aria-label="Open sidebar"]') ||
            doc.querySelector('[data-testid="collapsedControl"]') ||
            doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
        const closeBtn =
            doc.querySelector('button[aria-label="Close sidebar"]') ||
            doc.querySelector('[data-testid="stSidebarCollapseButton"]');

        if ("{action}" === "open" && openBtn) {{
            openBtn.click();
        }}
        if ("{action}" === "close" && closeBtn) {{
            closeBtn.click();
        }}
        </script>
        """,
        height=0,
    )

# ----------------------------------------
# LANDING / ENTRY PAGE
# ----------------------------------------
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    render_landing_page()
    st.stop()

# ---- User has entered the app ----

# ── Onboarding overlay (early-return pattern: blocks all content until dismissed)

# Investor profile question keys
_question_keys = [
    "q_exp", "q_goal", "q_horizon", "q_safety",
    "q_r1", "q_r2", "q_r3", "q_drop",
    "q_env_1", "q_env_2", "q_soc_1", "q_soc_2", "q_gov_1", "q_gov_2",
    "q_esg_imp", "q_esg_att",
]
# NOTE: do NOT pre-initialise question keys to None.
# st.radio with index=None manages its own session_state entry.
# Pre-setting a key to None blocks the radio from writing the user's answer.
if "profile_saved" not in st.session_state:
    st.session_state["profile_saved"] = False

# ── Auto-backup + restore answers on every rerun ────────────────────────────
# Runs before any widget is rendered so answers survive ALL reruns — not just
# nav button clicks.  Two-way sync:
#   • live value exists  → write to backup  (keeps backup fresh)
#   • live value is None → restore from backup (widget was cleared by Streamlit)
for _k in _question_keys:
    _bk = f"_ans_{_k}"
    _live_val = st.session_state.get(_k)
    if _live_val is not None:
        st.session_state[_bk] = _live_val          # keep backup fresh
    elif _bk in st.session_state:
        st.session_state[_k] = st.session_state[_bk]  # restore if cleared

# ----------------------------------------
# SIDEBAR
# ----------------------------------------

esg_filtered_df, esg_scores_df, esg_lookup = load_esg_data_from_path(ESG_DEFAULT_SOURCE)

# Scoring (from Investor Profile tab)
_abcd = {"A": 1, "B": 2, "C": 3, "D": 4}
_ab   = {"A": 1, "B": 4}

profile_complete = (
    st.session_state.get("profile_complete", False)
    or all(st.session_state.get(k) is not None for k in _question_keys)
)

if profile_complete:
    render_sidebar_logo()
else:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"] {
            margin-left: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if profile_complete:
    # If pre-computed values exist (from Save Profile), use them directly.
    # Otherwise, recalculate from the question keys (which were restored
    # from backups at the top of the script).
    if "gamma" in st.session_state:
        gamma = st.session_state["gamma"]
        lambda_raw_avg = st.session_state["lambda_val"]
        lambda_esg = (lambda_raw_avg - 1) / 300
    else:
        risk_scores = [
            _abcd[st.session_state["q_exp"][0]],
            _abcd[st.session_state["q_goal"][0]],
            _abcd[st.session_state["q_horizon"][0]],
            _abcd[st.session_state["q_safety"][0]],
            _ab[st.session_state["q_r1"][0]],
            _ab[st.session_state["q_r2"][0]],
            _ab[st.session_state["q_r3"][0]],
            _abcd[st.session_state["q_drop"][0]],
        ]
        risk_avg = sum(risk_scores) / len(risk_scores)
        gamma = 10 - 2 * risk_avg
        gamma = max(2.0, min(8.0, gamma))

        esg_scores_list = [
            _abcd[st.session_state["q_esg_imp"][0]],
            _abcd[st.session_state["q_esg_att"][0]],
        ]
        esg_avg = sum(esg_scores_list) / len(esg_scores_list)
        lambda_raw_avg = esg_avg
        lambda_esg = (esg_avg - 1) / 300
else:
    gamma = 5.0
    lambda_raw_avg = 2.5
    lambda_esg = (lambda_raw_avg - 1) / 300

if profile_complete:
    if "w_e" in st.session_state:
        w_e = st.session_state["w_e"]
        w_s = st.session_state["w_s"]
        w_g = st.session_state["w_g"]
        # Normalise to portfolio weights (sum to 1)
        _total_w = w_e + w_s + w_g
        if _total_w > 0:
            w_e, w_s, w_g = w_e / _total_w, w_s / _total_w, w_g / _total_w
        else:
            w_e, w_s, w_g = (1/3, 1/3, 1/3)
    else:
        _likert_to_int = lambda v: int(str(v).split(" ", 1)[0])
        e_score = (_likert_to_int(st.session_state["q_env_1"]) + _likert_to_int(st.session_state["q_env_2"])) / 2
        s_score = (_likert_to_int(st.session_state["q_soc_1"]) + _likert_to_int(st.session_state["q_soc_2"])) / 2
        g_score = (_likert_to_int(st.session_state["q_gov_1"]) + _likert_to_int(st.session_state["q_gov_2"])) / 2
        total_esg_pref = e_score + s_score + g_score
        if total_esg_pref > 0:
            w_e = e_score / total_esg_pref
            w_s = s_score / total_esg_pref
            w_g = g_score / total_esg_pref
        else:
            w_e, w_s, w_g = (1/3, 1/3, 1/3)
else:
    w_e, w_s, w_g = (1/3, 1/3, 1/3)

if "show_onboarding" not in st.session_state:
    st.session_state["show_onboarding"] = True

if profile_complete:
    _sync_sidebar_visibility(True)
    render_hero()
else:
    _sync_sidebar_visibility(False)
    if st.session_state["show_onboarding"]:
        render_onboarding_overlay()
        st.stop()

if "manual_esg_mode" not in st.session_state:
    st.session_state["manual_esg_mode"] = False

sin_stock_exclusions = []
selected_tickers = []
asset_profiles = {}
asset_esg_lookup = {}
lookback_years = 3.0
investment_amount = 0.0
risk_free_rate_pct = 2.0
run_button = False

if profile_complete:
    sin_stock_exclusions = st.sidebar.multiselect(
        "Exclude these sin industries",
        SIN_INDUSTRIES,
        default=[],
    )

    st.sidebar.header("Asset Inputs")
    manual_esg_mode = st.session_state["manual_esg_mode"]
    asset_count = st.sidebar.slider("Number of assets", min_value=2, max_value=10, value=2, step=1)

    for asset_idx in range(asset_count):
        asset_label = f"Asset {asset_idx + 1}"
        ticker = st.sidebar.text_input(
            f"Ticker for {asset_label}",
            value="",
            placeholder="e.g. AAPL" if asset_idx == 0 else "e.g. MSFT",
            key=f"ticker_{asset_idx + 1}",
        ).strip().upper()
        selected_tickers.append(ticker)
        profile = fetch_ticker_profile(ticker)
        asset_profiles[ticker] = profile
        esg_data = None
        for key in _ticker_variants(ticker):
            esg_data = esg_lookup.get(key)
            if esg_data is not None:
                break
        if esg_data is not None:
            esg_data = dict(esg_data)
            esg_data["ESG"] = weighted_esg(esg_data["E"], esg_data["S"], esg_data["G"], w_e, w_s, w_g)
        if manual_esg_mode or esg_data is None or pd.isna(esg_data.get("E")) or pd.isna(esg_data.get("S")) or pd.isna(esg_data.get("G")):
            if manual_esg_mode:
                st.sidebar.caption(f"Manual ESG input mode enabled for {ticker or asset_label}.")
            else:
                st.sidebar.caption(f"ESG data unavailable for {ticker or asset_label}. Enter E, S, G manually.")
            e_manual = st.sidebar.slider(f"E score for {ticker or asset_label}", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"e_manual_{asset_idx + 1}")
            s_manual = st.sidebar.slider(f"S score for {ticker or asset_label}", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"s_manual_{asset_idx + 1}")
            g_manual = st.sidebar.slider(f"G score for {ticker or asset_label}", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"g_manual_{asset_idx + 1}")
            esg_data = {
                "E": e_manual,
                "S": s_manual,
                "G": g_manual,
                "ESG": weighted_esg(e_manual, s_manual, g_manual, w_e, w_s, w_g),
            }
        asset_esg_lookup[ticker] = esg_data
        if ticker:
            render_sidebar_company_profile(profile, sin_stock_exclusions, esg_data)

    lookback_years = st.sidebar.slider(
        "Historical lookback period (years)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=1.0,
    )
period = f"{int(lookback_years)}y"
tickers = [ticker for ticker in selected_tickers if ticker]
asset_profiles = {ticker: asset_profiles.get(ticker) for ticker in tickers}
asset_esg_lookup = {ticker: asset_esg_lookup.get(ticker) for ticker in tickers}
asset_esg_scores = pd.Series({ticker: (asset_esg_lookup[ticker]["ESG"] if asset_esg_lookup.get(ticker) is not None else np.nan) for ticker in tickers}, dtype=float)

if profile_complete:
    st.sidebar.markdown("Total amount to invest (optional)")
    _amount_prefix_col, _amount_input_col = st.sidebar.columns([0.16, 0.84], gap="small")
    with _amount_prefix_col:
        st.markdown('<div class="currency-prefix">$</div>', unsafe_allow_html=True)
    with _amount_input_col:
        investment_amount = st.number_input(
            "Total amount to invest",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
            label_visibility="collapsed",
        )
    risk_free_rate_pct = st.sidebar.slider("Risk-free rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_free_rate = risk_free_rate_pct / 100

if profile_complete:
    manual_button = st.sidebar.button("Manual ESG Input")
    if manual_button:
        st.session_state["manual_esg_mode"] = not st.session_state["manual_esg_mode"]
    st.sidebar.caption(f"Manual ESG Input: {'On' if st.session_state['manual_esg_mode'] else 'Off'}")

# FEATURE 2 — Investor Profile Card in Sidebar
if profile_complete:
    render_sidebar_profile_card(gamma, lambda_raw_avg, w_e, w_s, w_g)

if profile_complete:
    run_button = st.sidebar.button("Run portfolio optimisation")
if "run_optimisation" not in st.session_state:
    st.session_state["run_optimisation"] = False
if run_button:
    st.session_state["run_optimisation"] = True

# ========================================
# MAIN OUTPUT
# ========================================

if profile_complete:
    tab0, tab1, tab2, tab3, tab5 = st.tabs([
        " Investor Profile",
        " Dashboard",
        " Portfolio Analysis",
        " ESG Breakdown",
        " Compatibility & Alternatives",
    ])
else:
    tab0 = st.tabs([" Investor Profile"])[0]
    tab1 = tab0


def _jump_to_tab(tab_index: int):
    components.html(
        f"""
        <script>
        const tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] button[role="tab"]');
        const target = tabs[{tab_index}];
        if (target) {{
            target.click();
        }}
        </script>
        """,
        height=0,
    )

# ── Questionnaire wizard state ──────────────────────────────────────────────
if "q_step" not in st.session_state:
    st.session_state["q_step"] = 1
# Clamp: never allow q_step outside 1–3
st.session_state["q_step"] = max(1, min(3, int(st.session_state["q_step"])))

_STEP_KEYS = {
    1: ["q_exp", "q_goal", "q_horizon", "q_safety"],
    2: ["q_r1", "q_r2", "q_r3", "q_drop"],
    3: ["q_env_1", "q_env_2", "q_soc_1", "q_soc_2", "q_gov_1", "q_gov_2", "q_esg_imp", "q_esg_att"],
}
_STEP_TITLES  = {1: "About You", 2: "Risk Preference", 3: "ESG Preferences"}
_STEP_ICONS   = {1: "👤", 2: "📊", 3: "🌱"}
_STEP_DESCS   = {
    1: "Tell us a bit about yourself and your investment goals.",
    2: "Help us understand how you feel about financial risk.",
    3: "Share your sustainability values so we can match your portfolio.",
}

# Tab 0 - Investor Profile
with tab0:

    st.markdown("""
    <style>
      @keyframes stepFadeIn {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      .wizard-section-head {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.07em;
        color: #1a6b3c; text-transform: uppercase;
        margin: 1.4rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0f0e6;
      }
      .wizard-warn {
        background: #fff7ed; border: 1px solid #fed7aa;
        border-left: 4px solid #f59e0b; border-radius: 10px;
        padding: 0.65rem 1rem; font-size: 0.88rem; color: #92400e;
        margin: 0.6rem 0 0.8rem 0;
      }
      .wizard-success {
        background: linear-gradient(135deg,#e8f5ee,#f0faf4);
        border: 1px solid #a7d7bc; border-radius: 16px;
        padding: 1.4rem 1.8rem; text-align: center;
        animation: stepFadeIn 0.4s ease both; margin-bottom: 1.4rem;
      }
      .step-dot-done   { background:#1a6b3c !important; color:#fff !important; box-shadow:0 4px 12px rgba(26,107,60,0.28); }
      .step-dot-active { background:linear-gradient(135deg,#1a6b3c,#2d8a4e) !important; color:#fff !important; box-shadow:0 4px 16px rgba(26,107,60,0.38); }
      .step-dot-pend   { background:#e8f0ea !important; color:#7a9a82 !important; border:2px solid #c5d9cc; }
    </style>
    """, unsafe_allow_html=True)

    likert_5 = ["1 - Strongly disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly agree"]

    # ── COMPLETED VIEW — shown instead of wizard once all 16 answered ─────────
    if profile_complete:
        st.markdown("""
        <div class="wizard-success">
          <div style="font-size:2.6rem;margin-bottom:0.4rem;">✅</div>
          <div style="font-size:1.15rem;font-weight:800;color:#1a6b3c;">Profile saved successfully</div>
          <div style="font-size:0.92rem;color:#5a7a63;margin-top:0.35rem;">
            Your investor profile is complete. Head to the <strong>Dashboard</strong> tab and hit
            <strong>Run portfolio optimisation</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        _pc1, _pc2, _pc3, _pc4 = st.columns(4)
        render_metric_card(_pc1, "Risk attitude",          classify_risk(gamma))
        render_metric_card(_pc2, "Sustainability profile", classify_esg(lambda_raw_avg))
        render_metric_card(_pc3, "Risk aversion (γ)",      f"{gamma:.2f}")
        render_metric_card(_pc4, "ESG preference (λ)",     f"{lambda_raw_avg:.2f} / 4")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _, _edit_col, _ = st.columns([1, 1, 1])
        with _edit_col:
            if st.button("✏️  Edit Profile", key="q_edit", use_container_width=True):
                for _k in _question_keys:
                    st.session_state.pop(_k, None)
                    st.session_state.pop(f"_ans_{_k}", None)
                st.session_state["profile_saved"] = False
                st.session_state["profile_complete"] = False
                st.session_state.pop("gamma", None)
                st.session_state.pop("lambda_val", None)
                st.session_state.pop("w_e", None)
                st.session_state.pop("w_s", None)
                st.session_state.pop("w_g", None)
                st.session_state["q_step"] = 1
                st.rerun()

    else:
        # ── WIZARD VIEW ────────────────────────────────────────────────────────
        _step       = st.session_state["q_step"]
        _total      = 3
        _answered   = sum(
            1 for k in _question_keys
            if st.session_state.get(k) is not None or st.session_state.get(f"_ans_{k}") is not None
        )

        # Progress bar (overall)
        st.markdown(
            f'<div style="font-size:0.82rem;color:#5a7a63;margin-bottom:0.5rem;">'
            f'Overall progress: <strong>{_answered} / {len(_question_keys)}</strong> questions answered</div>',
            unsafe_allow_html=True,
        )
        st.progress(_answered / len(_question_keys))

        # Step indicator dots
        _dot_html = '<div style="display:flex;align-items:center;gap:0;margin:1rem 0 1.4rem 0;">'
        _step_info = {1: "About You", 2: "Risk Preference", 3: "ESG Preferences"}
        for _i in range(1, _total + 1):
            _cls = "step-dot-done" if _i < _step else ("step-dot-active" if _i == _step else "step-dot-pend")
            _lbl_color = "#1a6b3c" if _i == _step else "#5a7a63"
            _content = "✓" if _i < _step else str(_i)
            _dot_html += (
                f'<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">'
                f'<div class="{_cls}" style="width:34px;height:34px;border-radius:50%;'
                f'display:flex;align-items:center;justify-content:center;font-size:0.82rem;font-weight:700;">'
                f'{_content}</div>'
                f'<div style="font-size:0.7rem;font-weight:600;color:{_lbl_color};margin-top:0.3rem;'
                f'letter-spacing:0.03em;">{_step_info[_i]}</div></div>'
            )
            if _i < _total:
                _line_bg = "linear-gradient(90deg,#1a6b3c,#4caf7d)" if _i < _step else "#dde8e0"
                _dot_html += f'<div style="flex:1;height:3px;border-radius:999px;background:{_line_bg};margin-bottom:1.2rem;"></div>'
        _dot_html += '</div>'
        st.markdown(_dot_html, unsafe_allow_html=True)

        # Step header card
        _icons = {1: "👤", 2: "📊", 3: "🌱"}
        _descs = {
            1: "Tell us a bit about yourself and your investment goals.",
            2: "Help us understand how you feel about financial risk.",
            3: "Share your sustainability values so we can match your portfolio.",
        }
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#f0f7f2,#e8f5ee);'
            f'border:1px solid #c5d9cc;border-radius:16px;padding:1.2rem 1.6rem;'
            f'margin-bottom:1.2rem;animation:stepFadeIn 0.3s ease both;">'
            f'<span style="font-size:1.8rem;">{_icons[_step]}</span>'
            f'<span style="font-size:1.2rem;font-weight:800;color:#1a2e1f;margin-left:0.6rem;">'
            f'{_step_info[_step]}</span>'
            f'<div style="font-size:0.92rem;color:#5a7a63;margin-top:0.4rem;">{_descs[_step]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _wl, _wc, _wr = st.columns([0.08, 0.84, 0.08])
        with _wc:

            # ── Step 1 ──────────────────────────────────────────────────────
            if _step == 1:
                render_sage("Your answers here shape your risk aversion score (γ). A higher γ means you prefer safer, steadier returns over chasing higher but volatile gains.")
                st.radio("1. How experienced are you with investing?",
                    ["A: No experience", "B: Basic knowledge (e.g. savings, ETFs)",
                     "C: Some experience (stocks, funds)", "D: Advanced (active trading, portfolio management)"],
                    key="q_exp", index=None)
                st.radio("2. What is your primary investment objective?",
                    ["A: Preserve my capital", "B: Generate stable income",
                     "C: Grow my wealth steadily", "D: Maximise long-term returns"],
                    key="q_goal", index=None)
                st.radio("3. When do you expect to need this money?",
                    ["A: Less than 1 year", "B: 1–3 years", "C: 3–7 years", "D: More than 7 years"],
                    key="q_horizon", index=None)
                st.radio("4. Do you have emergency savings?",
                    ["A: None", "B: Less than 3 months of expenses", "C: 3–6 months", "D: More than 6 months"],
                    key="q_safety", index=None)

            # ── Step 2 ──────────────────────────────────────────────────────
            elif _step == 2:
                render_sage("These questions determine how much of your portfolio goes into risky assets vs the risk-free rate. The more risk-tolerant you are, the higher your allocation to stocks.")
                st.markdown('<div class="wizard-section-head">Choose the portfolio you prefer</div>', unsafe_allow_html=True)
                st.radio("Round 1 — which do you prefer?",
                    ["A: Return 3%, Risk 5%", "B: Return 8%, Risk 12%"],
                    key="q_r1", index=None)
                st.radio("Round 2",
                    ["A: Return 6%, Risk 7%", "B: Return 10%, Risk 18%"],
                    key="q_r2", index=None)
                st.radio("Round 3",
                    ["A: Return 7%, Risk 9%", "B: Return 13%, Risk 22%"],
                    key="q_r3", index=None)
                st.markdown('<div class="wizard-section-head">Market scenario</div>', unsafe_allow_html=True)
                st.radio("If the stock market suddenly dropped by 30%, how would you react?",
                    ["A: Sell all my investments immediately",
                     "B: Sell some of my investments to reduce risk",
                     "C: Do nothing", "D: Invest more"],
                    key="q_drop", index=None)

            # ── Step 3 ──────────────────────────────────────────────────────
            elif _step == 3:
                render_sage("Your ESG score (λ) controls how much sustainability influences your portfolio. A higher λ shifts weights toward greener assets — even if it costs some Sharpe ratio points.")
                st.caption("Questions 9 – 16  ·  Rate each statement from 1 (strongly disagree) to 5 (strongly agree).")
                st.markdown('<div class="wizard-section-head">🌍 Environmental</div>', unsafe_allow_html=True)
                st.radio("9. I prefer companies with lower carbon emissions.",
                         likert_5, key="q_env_1", index=None)
                st.radio("10. I am willing to accept slightly lower returns for stronger environmental performance.",
                         likert_5, key="q_env_2", index=None)
                st.markdown('<div class="wizard-section-head">🤝 Social</div>', unsafe_allow_html=True)
                st.radio("11. I care about labour standards and employee welfare.",
                         likert_5, key="q_soc_1", index=None)
                st.radio("12. I prefer firms with strong diversity and human rights practices.",
                         likert_5, key="q_soc_2", index=None)
                st.markdown('<div class="wizard-section-head">🏛 Governance</div>', unsafe_allow_html=True)
                st.radio("13. Good board oversight and transparency are very important to me.",
                         likert_5, key="q_gov_1", index=None)
                st.radio("14. I prefer firms with strong shareholder rights and ethical management.",
                         likert_5, key="q_gov_2", index=None)
                st.markdown('<div class="wizard-section-head">⚖️ Your ESG Stance</div>', unsafe_allow_html=True)
                st.radio("15. Which best reflects your attitude toward ESG vs financial returns?",
                    ["A: I focus only on financial returns, without considering ESG factors.",
                     "B: ESG factors matter to me, but do not really influence my investment decisions.",
                     "C: ESG considerations play an important role alongside financial returns.",
                     "D: ESG impact is my top priority and I am willing to give up some return."],
                    key="q_esg_imp", index=None)
                st.radio("16. To what extent does a higher ESG score make an investment more attractive?",
                    ["A: Never — I would only consider ESG if financial performance is very strong.",
                     "B: Slightly — I would include it if it does not negatively impact returns.",
                     "C: Moderately — I value ESG and aim to invest in high-ESG-score companies.",
                     "D: Significantly — I always prioritise ESG, even if returns are lower."],
                    key="q_esg_att", index=None)

            # ── Per-step count + warning ────────────────────────────────────
            _step_keys_now  = _STEP_KEYS[_step]
            _step_complete  = all(st.session_state.get(k) is not None for k in _step_keys_now)
            _step_n_done    = sum(1 for k in _step_keys_now if st.session_state.get(k) is not None)

            st.markdown(
                f'<div style="font-size:0.79rem;color:#5a7a63;margin-top:0.4rem;">'
                f'{_step_n_done} / {len(_step_keys_now)} answered on this step</div>',
                unsafe_allow_html=True,
            )
            if not _step_complete:
                st.markdown(
                    '<div class="wizard-warn">⚠️ Answer all questions on this step before continuing.</div>',
                    unsafe_allow_html=True,
                )

            # ── helper: back up current step answers so they survive rerun ──
            def _backup_step_answers():
                """Copy current step widget values to persistent backup keys."""
                for _bk in _STEP_KEYS[_step]:
                    _val = st.session_state.get(_bk)
                    if _val is not None:
                        st.session_state[f"_ans_{_bk}"] = _val

            # ── Nav buttons ─────────────────────────────────────────────────
            _nav_l, _nav_r = st.columns([1, 1])
            with _nav_l:
                if _step > 1:
                    if st.button("← Back", key="q_back", use_container_width=True):
                        _backup_step_answers()
                        st.session_state["q_step"] = _step - 1
                        st.rerun()
            with _nav_r:
                if _step < _total:
                    if st.button("Next →", key="q_next", use_container_width=True):
                        if _step_complete:
                            _backup_step_answers()
                            st.session_state["q_step"] = _step + 1
                            st.rerun()
                        else:
                            st.warning("Please answer every question on this step first.")
                else:
                    # Step 3: Save Profile
                    if st.button("Save Profile ✓", key="q_save", use_container_width=True):
                        if _step_complete:
                            _backup_step_answers()

                            # -- Collect all 16 answers (current step from widgets,
                            #    earlier steps from backups) --
                            def _get_answer(key):
                                """Return the answer for a question key, preferring
                                the live widget value, falling back to the backup."""
                                v = st.session_state.get(key)
                                if v is not None:
                                    return v
                                return st.session_state.get(f"_ans_{key}")

                            # -- Compute gamma from risk questions --
                            _risk_keys = ["q_r1", "q_r2", "q_r3", "q_drop"]
                            # Risk questions use A/B or A/B/C/D options.
                            # The scoring maps used at the top of the file:
                            #   _ab   (q_r1, q_r2, q_r3) : A -> 1, B -> 4
                            #   _abcd (q_drop)            : A -> 1, B -> 2, C -> 3, D -> 4
                            _risk_score_map = {
                                "q_r1": _ab, "q_r2": _ab, "q_r3": _ab,
                                "q_drop": _abcd,
                            }
                            _risk_scores = [
                                _risk_score_map[rk][_get_answer(rk)[0]]
                                for rk in _risk_keys
                            ]
                            _gamma = 10 - 2 * (sum(_risk_scores) / len(_risk_scores))
                            _gamma = max(2.0, min(8.0, _gamma))

                            # -- Compute lambda from ESG attitude questions --
                            _esg_att_keys = ["q_esg_imp", "q_esg_att"]
                            _esg_att_scores = [
                                _abcd[_get_answer(ek)[0]]
                                for ek in _esg_att_keys
                            ]
                            _lambda_val = sum(_esg_att_scores) / len(_esg_att_scores)

                            # -- Compute ESG pillar weights --
                            _likert_int = lambda v: int(str(v).split(" ", 1)[0])
                            _w_e = (_likert_int(_get_answer("q_env_1")) + _likert_int(_get_answer("q_env_2"))) / 2
                            _w_s = (_likert_int(_get_answer("q_soc_1")) + _likert_int(_get_answer("q_soc_2"))) / 2
                            _w_g = (_likert_int(_get_answer("q_gov_1")) + _likert_int(_get_answer("q_gov_2"))) / 2

                            # -- Store computed values in session_state --
                            # These keys are read by the scoring block at the top
                            # of the script on subsequent reruns.
                            st.session_state["gamma"] = _gamma
                            st.session_state["lambda_val"] = _lambda_val
                            st.session_state["w_e"] = _w_e
                            st.session_state["w_s"] = _w_s
                            st.session_state["w_g"] = _w_g

                            st.session_state["profile_saved"] = True
                            st.session_state["profile_complete"] = True
                            st.success("Profile saved!")
                            st.rerun()
                        else:
                            st.warning("Please answer all 8 questions on this step first.")

if st.session_state.get("run_optimisation", False):
    if not profile_complete:
        with tab1:
            st.error("Please complete all Investor Profile questions first.")
    elif len(tickers) < 2:
        with tab1:
            st.error("Please enter at least two tickers.")
    elif len(set(tickers)) != len(tickers):
        with tab1:
            st.error("Please choose different tickers.")
    elif any(asset_profiles.get(ticker) and asset_profiles[ticker]["industry"] in sin_stock_exclusions for ticker in tickers):
        blocked_ticker = next(ticker for ticker in tickers if asset_profiles.get(ticker) and asset_profiles[ticker]["industry"] in sin_stock_exclusions)
        with tab1:
            st.error(f"{blocked_ticker} belongs to an excluded sin industry: {asset_profiles[blocked_ticker]['industry']}.")
    else:
        try:
            market_data = fetch_market_data(tickers, period=period)
            mean_returns = market_data["mean_returns"].loc[tickers]
            covariance = market_data["covariance"].loc[tickers, tickers]
            volatilities = market_data["volatilities"].loc[tickers]
            esg_scores = asset_esg_scores.loc[tickers]
            corr_matrix = market_data["correlation_matrix"].loc[tickers, tickers]
            if len(tickers) == 2:
                avg_corr = float(corr_matrix.iloc[0, 1])
            else:
                corr_values = corr_matrix.to_numpy()
                upper = corr_values[np.triu_indices_from(corr_values, k=1)]
                avg_corr = float(np.nanmean(upper)) if len(upper) else np.nan
            market_data["avg_correlation"] = avg_corr

            df = build_portfolio_table(
                expected_returns=mean_returns.values,
                covariance_matrix=covariance.values,
                rf=risk_free_rate,
                esg_scores=esg_scores.values,
                gamma=gamma,
                lambda_esg=lambda_esg,
                tickers=tickers,
            )

            recommended = select_recommended_portfolio(df)
            tangency = select_max_sharpe_portfolio(df)
            complete_portfolio = build_complete_portfolio(tangency, risk_free_rate, gamma)
            recommended_weights = np.array([float(recommended.get('Risky_Weights', {}).get(ticker, 0.0)) for ticker in tickers], dtype=float)
            tangency_weights = get_weight_vector(tangency, tickers)

            esg_scores_weighted = esg_scores_df.copy()
            esg_scores_weighted["ESG"] = weighted_esg(
                esg_scores_weighted["E"],
                esg_scores_weighted["S"],
                esg_scores_weighted["G"],
                w_e, w_s, w_g,
            )
            alternatives_warning = None
            try:
                universe_returns = fetch_universe_returns(esg_scores_df["ticker"].tolist(), period)
                candidates = esg_scores_weighted.merge(universe_returns, on="ticker", how="inner")
                candidates = candidates.dropna(subset=["Expected_Return", "ESG"])
                candidates = candidates[~candidates["ticker"].isin(tickers)]
            except Exception:
                universe_returns = pd.DataFrame(columns=["ticker", "Expected_Return", "Volatility"])
                candidates = pd.DataFrame(columns=list(esg_scores_weighted.columns) + ["Expected_Return", "Volatility"])
                alternatives_warning = (
                    "Yahoo Finance did not return the broader universe data on this run. "
                    "Your main portfolio was still computed from the selected tickers."
                )

            dim_weights = {"E": w_e, "S": w_s, "G": w_g}
            primary_dim = max(dim_weights, key=dim_weights.get)
            primary_focus_labels = {"E": "Environmental (E)", "S": "Social (S)", "G": "Governance (G)"}
            primary_focus_colors = {"E": "#166534", "S": "#1e40af", "G": "#6b21a8"}
            primary_focus = primary_focus_labels[primary_dim]
            focus_color = primary_focus_colors[primary_dim]

            def compute_profile_alignment(esg_data, w_e, w_s, w_g):
                if esg_data is None:
                    return (None,) * 7
                E = esg_data.get("E") or 0
                S = esg_data.get("S") or 0
                G = esg_data.get("G") or 0
                if pd.isna(E) or pd.isna(S) or pd.isna(G):
                    return (None,) * 7
                E, S, G = float(E), float(S), float(G)
                weighted_quality = (w_e * E + w_s * S + w_g * G)
                total_esg = E + S + G
                if total_esg > 0:
                    se, ss, sg = E / total_esg, S / total_esg, G / total_esg
                else:
                    se = ss = sg = 1 / 3
                dot = w_e * se + w_s * ss + w_g * sg
                mag_inv = np.sqrt(w_e ** 2 + w_s ** 2 + w_g ** 2) or 1e-9
                mag_stock = np.sqrt(se ** 2 + ss ** 2 + sg ** 2) or 1e-9
                profile_match = (dot / (mag_inv * mag_stock)) * 100
                e_contrib = w_e * E
                s_contrib = w_s * S
                g_contrib = w_g * G
                gaps = (w_e - E, w_s - S, w_g - G)
                composite = 0.6 * weighted_quality + 0.4 * profile_match
                return composite, profile_match, weighted_quality, e_contrib, s_contrib, g_contrib, gaps

            def alignment_badge(score):
                if score is None:
                    return "No Data", "#6b7280"
                if score >= 70:
                    return "Strong Alignment", "#166534"
                elif score >= 45:
                    return "Moderate Alignment", "#92400e"
                return "Weak Alignment", "#991b1b"

            asset_alignment = {ticker: compute_profile_alignment(asset_esg_lookup.get(ticker), w_e, w_s, w_g) for ticker in tickers}
            compatibility_by_asset = [
                {"ticker": ticker, "composite": asset_alignment[ticker][0], "profile_match": asset_alignment[ticker][1], "quality": asset_alignment[ticker][2]}
                for ticker in tickers
            ]

            fig0 = make_price_history_figure(market_data["prices"], tickers)
            fig1 = make_frontier_figure(df, tangency, recommended, tickers, mean_returns, volatilities)
            fig2 = make_cml_figure(df, tangency, complete_portfolio, risk_free_rate, tickers, mean_returns, volatilities)
            fig3 = make_esg_tradeoff_figure(df, recommended)
            fig4 = make_esg_efficient_frontier_figure(df, tangency, recommended)

            compat = compute_portfolio_compatibility(
                gamma=gamma,
                lambda_raw_avg=lambda_raw_avg,
                w_e=w_e, w_s=w_s, w_g=w_g,
                recommended=recommended,
                esg_data_list=[asset_esg_lookup.get(ticker) for ticker in tickers],
                weights=recommended_weights,
            )

            def _weight_bar_cell(text_value, tone="tan"):
                try:
                    raw_pct = float(str(text_value).replace("%", "").strip())
                except Exception:
                    return text_value
                pct = min(100.0, abs(raw_pct))
                is_negative = raw_pct < 0
                if is_negative:
                    fill = "#9a5b63"
                    bg = "#ead9dd"
                    border = "#d9c2c7"
                elif tone == "tan":
                    fill = "#7f9bc2"
                    bg = "#d9e3f1"
                    border = "#cfdae6"
                else:
                    fill = "#6b847a"
                    bg = "#d8e3de"
                    border = "#cfdae6"
                return (
                    f'<div style="display:flex;align-items:center;gap:0.45rem;">'
                    f'<div style="width:78px;height:10px;border-radius:7px;background:{bg};overflow:hidden;border:1px solid {border};">'
                    f'<div style="width:{pct:.2f}%;height:100%;background:{fill};"></div>'
                    f'</div>'
                    f'<span>{raw_pct:.2f}%</span>'
                    f'</div>'
                )

            def _small_stat_card(label, value):
                return (
                    f'<div class="metric-card" style="min-height:88px;padding:0.58rem 0.72rem;">'
                    f'<div class="metric-card-label">{label}</div>'
                    f'<div class="metric-card-value" style="font-size:1.78rem;">{value}</div>'
                    f'</div>'
                )

            def _build_weight_table(portfolio, weight_label, amount_label):
                weights = portfolio.get("Weights", {})
                table_df = pd.DataFrame([
                    {"Asset": ticker, weight_label: weights.get(ticker, 0.0), amount_label: investment_amount * weights.get(ticker, 0.0)}
                    for ticker in tickers
                ])
                table_df[weight_label] = table_df[weight_label].map(lambda x: f"{x * 100:.2f}%")
                table_df[amount_label] = table_df[amount_label].map(lambda x: f"{x:,.2f}")
                return table_df

# --------------------------------------------------
# TAB 1 - Dashboard
# --------------------------------------------------
            with tab1:
                render_section_title("Investor Profile")
                c1, c2, c3, c4 = st.columns(4)
                render_metric_card(c1, "Risk attitude", classify_risk(gamma))
                render_metric_card(c2, "Sustainability profile", classify_esg(lambda_raw_avg))
                render_metric_card(c3, "Risk aversion score", f"{gamma:.2f}")
                render_metric_card(c4, "ESG preference score", f"{lambda_raw_avg:.2f} / 4")

                render_section_title("Your Portfolio at a Glance")
                render_sage("This is your recommended portfolio after balancing your risk profile and ESG values. The weights show how to split your investment across your chosen assets and the risk-free rate.")
                p1, p2, p3, p4, p5 = st.columns(5)
                render_metric_card(p1, "Expected return", f"{recommended['Expected_Return']*100:.2f}%")
                render_metric_card(p2, "Volatility", f"{recommended['Risk_SD']*100:.2f}%")
                render_metric_card(p3, "ESG score", f"{recommended['ESG_Score']:.2f}")
                render_metric_card(p4, "Sharpe ratio", f"{recommended['Sharpe_Ratio']:.3f}")
                render_metric_card(p5, "ESG utility", f"{recommended['Utility']:.4f}")

                # FEATURE 1 — Animated ESG Score Gauge
                render_esg_gauge(recommended["ESG_Score"])

                render_green_cost_card(
                    sharpe_gap=tangency["Sharpe_Ratio"] - recommended["Sharpe_Ratio"],
                    esg_gain=recommended["ESG_Score"] - tangency["ESG_Score"],
                )

                if st.button("Next ", key="next_tab1"):
                    _jump_to_tab(2)

# --------------------------------------------------
# TAB 2 - Portfolio Analysis
# --------------------------------------------------
            with tab2:
                tangency_weights_df = _build_weight_table(tangency, "Tangency Weight", "Amount if fully invested in tangency portfolio")
                tangency_weights_view = tangency_weights_df.copy()
                tangency_weights_view["Tangency Weight"] = tangency_weights_view["Tangency Weight"].map(lambda x: _weight_bar_cell(x, "tan"))
                tan_table_html = style_table(tangency_weights_view).to_html(escape=False)
                md = pd.DataFrame({"Metric": ["Expected annual return", "Annual volatility", "ESG score used"]})
                for ticker in tickers:
                    md[ticker] = [
                        f"{mean_returns.loc[ticker] * 100:.2f}%",
                        f"{volatilities.loc[ticker] * 100:.2f}%",
                        f"{esg_scores.loc[ticker]:.2f}",
                    ]
                md_table_html = style_table(md).to_html()
                corr_label = "Correlation" if len(tickers) == 2 else "Avg correlation"

                render_section_title("Charts")
                render_sage("The efficient frontier shows every possible combination of your assets. The green diamond is your ESG-optimal portfolio — notice how it differs from the pure Sharpe maximum.")
                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    st.pyplot(fig0, width="stretch")
                with ch_col2:
                    st.pyplot(fig1, width="stretch")
                ch_col3, ch_col4 = st.columns(2)
                with ch_col3:
                    st.pyplot(fig2, width="stretch")



                top_left, top_right = st.columns([1.12, 1.35], gap="large")
                with top_left:
                    render_section_title("Market Data Summary")
                    st.markdown(
                        f"""
                        <div style="width:100%;background:#ffffff;border:1px solid #d7dee5;border-radius:16px;padding:0.6rem;box-shadow:0 8px 18px rgba(15,23,42,0.06);">
                          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0.7rem;align-items:start;width:100%;">
                            <div style="min-width:0;overflow-x:auto;">{md_table_html}</div>
                            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.5rem;">
                              {_small_stat_card(corr_label, f"{avg_corr:.3f}")}
                              {_small_stat_card("Risk-free", f"{risk_free_rate * 100:.2f}%")}
                            </div>
                          </div>
                        </div>

                        """,
                        unsafe_allow_html=True,
                    )
                with top_right:
                    render_section_title("Tangency Portfolio")
                    st.markdown(
                        f"""
                        <div style="width:100%;background:#ffffff;border:1px solid #d7dee5;border-radius:16px;padding:0.6rem;box-shadow:0 8px 18px rgba(15,23,42,0.06);">
                          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0.6rem;align-items:start;width:100%;">
                            <div style="min-width:0;overflow-x:auto;">{tan_table_html}</div>
                            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.5rem;">
                              {_small_stat_card("Expected return", f"{tangency['Expected_Return']*100:.2f}%")}
                              {_small_stat_card("Volatility", f"{tangency['Risk_SD']*100:.2f}%")}
                              {_small_stat_card("Sharpe ratio", f"{tangency['Sharpe_Ratio']:.3f}")}
                              {_small_stat_card("ESG score", f"{tangency['ESG_Score']:.2f}")}
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                recommended_weights_df = _build_weight_table(recommended, "Recommended Weight", "Amount")
                recommended_weights_df = pd.concat([recommended_weights_df, pd.DataFrame([{
                    "Asset": "Risk-free asset",
                    "Recommended Weight": f"{recommended['weight_risk_free'] * 100:.2f}%",
                    "Amount": f"{investment_amount * recommended['weight_risk_free']:,.2f}",
                }])], ignore_index=True)
                recommended_weights_view = recommended_weights_df.copy()
                recommended_weights_view["Recommended Weight"] = recommended_weights_view["Recommended Weight"].map(
                    lambda x: _weight_bar_cell(x, "rec") if "Risk-free" not in str(x) else x
                )
                rec_table_html = style_table(recommended_weights_view).to_html(escape=False)
                complete_weights_df = _build_weight_table(complete_portfolio, "Complete Portfolio Weight", "Amount")
                complete_weights_df = pd.concat([complete_weights_df, pd.DataFrame([{
                    "Asset": "Risk-free asset",
                    "Complete Portfolio Weight": f"{complete_portfolio['weight_risk_free'] * 100:.2f}%",
                    "Amount": f"{investment_amount * complete_portfolio['weight_risk_free']:,.2f}",
                }])], ignore_index=True)
                complete_weights_view = complete_weights_df.copy()
                complete_weights_view["Complete Portfolio Weight"] = complete_weights_view["Complete Portfolio Weight"].map(
                    lambda x: _weight_bar_cell(x, "rec") if "Risk-free" not in str(x) else x
                )
                comp_table_html = style_table(complete_weights_view).to_html(escape=False)
                rec_complete_left, rec_complete_right = st.columns([1, 1], gap="large")
                with rec_complete_left:
                    render_section_title("Recommended ESG Portfolio")
                    st.markdown(
                        f"""
                        <div style="width:100%;background:#ffffff;border:1px solid #d7dee5;border-radius:16px;padding:0.6rem;box-shadow:0 8px 18px rgba(15,23,42,0.06);">
                          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0.45rem;align-items:start;width:100%;">
                            <div style="min-width:0;overflow-x:auto;">{rec_table_html}</div>
                            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.5rem;">
                              {_small_stat_card("Expected return", f"{recommended['Expected_Return']*100:.2f}%")}
                              {_small_stat_card("Volatility", f"{recommended['Risk_SD']*100:.2f}%")}
                              {_small_stat_card("ESG score", f"{recommended['ESG_Score']:.2f}")}
                              {_small_stat_card("Sharpe ratio", f"{recommended['Sharpe_Ratio']:.3f}")}
                              <div style="grid-column:1 / span 2;">{_small_stat_card("ESG utility", f"{recommended['Utility']:.4f}")}</div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with rec_complete_right:
                    render_section_title("Complete Portfolio")
                    st.markdown(
                        f"""
                        <div style="width:100%;background:#ffffff;border:1px solid #d7dee5;border-radius:16px;padding:0.6rem;box-shadow:0 8px 18px rgba(15,23,42,0.06);">
                          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0.45rem;align-items:start;width:100%;">
                            <div style="min-width:0;overflow-x:auto;">{comp_table_html}</div>
                            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.5rem;">
                              {_small_stat_card("Tangency weight (y)", f"{complete_portfolio['y']:.3f}")}
                              {_small_stat_card("Risk-free weight", f"{complete_portfolio['weight_risk_free']:.3f}")}
                              {_small_stat_card("Expected return", f"{complete_portfolio['Expected_Return']*100:.2f}%")}
                              {_small_stat_card("Volatility", f"{complete_portfolio['Risk_SD']*100:.2f}%")}
                              <div style="grid-column:1 / span 2;">{_small_stat_card("Utility", f"{complete_portfolio['Utility']:.4f}")}</div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Recommended ESG objective: x'(mu - rf*1) - 0.5*gamma*x'Sigma x + lambda*s_bar, "
                        "with s_bar = (x's)/(x'1). Complete portfolio formulas: Expected Return = rf + x'(mu-rf*1), "
                        "Variance = x'Sigma x."
                    )

                render_section_title("Risky Portfolio Comparison")
                risky_compare_rows = [{"Metric": f"Weight in {ticker}", "Recommended ESG Portfolio": f"{recommended_weights[idx] * 100:.2f}%", "Tangency Portfolio": f"{tangency_weights[idx] * 100:.2f}%"} for idx, ticker in enumerate(tickers)]
                risky_compare_rows.extend([
                    {"Metric": "ESG score", "Recommended ESG Portfolio": f"{recommended['Risky_ESG_Score']:.2f}", "Tangency Portfolio": f"{tangency['ESG_Score']:.2f}"},
                    {"Metric": "ESG objective", "Recommended ESG Portfolio": f"{recommended['Utility']:.4f}", "Tangency Portfolio": "N/A"},
                    {"Metric": "Sharpe ratio", "Recommended ESG Portfolio": f"{recommended['Risky_Sharpe_Ratio']:.3f}", "Tangency Portfolio": f"{tangency['Sharpe_Ratio']:.3f}"},
                    {"Metric": "Expected return", "Recommended ESG Portfolio": f"{recommended['Risky_Expected_Return']*100:.2f}%", "Tangency Portfolio": f"{tangency['Expected_Return']*100:.2f}%"},
                    {"Metric": "Volatility", "Recommended ESG Portfolio": f"{recommended['Risky_Risk_SD']*100:.2f}%", "Tangency Portfolio": f"{tangency['Risk_SD']*100:.2f}%"},
                ])
                risky_compare_df = pd.DataFrame(risky_compare_rows)
                render_table(risky_compare_df)

                render_section_title("Complete Portfolio Comparison")
                if hasattr(ui, "render_complete_portfolio_comparison"):
                    ui.render_complete_portfolio_comparison(tangency, complete_portfolio, tickers)

                render_section_title("What-if Weight Simulator")
                st.caption(
                    "Drag the slider to explore how different allocations affect your portfolio metrics  "
                    "updates instantly without re-running the optimisation."
                )
                if len(tickers) == 2:
                    whatif_w1 = st.slider(
                        f"Weight in {tickers[0]} (%)  remainder goes to {tickers[1]}",
                        min_value=0,
                        max_value=100,
                        value=int(recommended_weights[0] * 100),
                        step=1,
                        key="whatif_w1",
                    )
                    whatif_weights = np.array([whatif_w1 / 100, 1 - (whatif_w1 / 100)], dtype=float)
                    split_label = f"{whatif_w1}% / {100 - whatif_w1}%"
                else:
                    default_ticker = tickers[int(np.argmax(recommended_weights))]
                    whatif_ticker = st.selectbox("Asset to adjust", tickers, index=tickers.index(default_ticker))
                    whatif_weight = st.slider(
                        f"Weight in {whatif_ticker} (%)",
                        min_value=0,
                        max_value=100,
                        value=int(float(recommended.get(f"Weight_{whatif_ticker}", 0.0)) * 100),
                        step=1,
                        key="whatif_weight_n",
                    )
                    whatif_weights = recommended_weights.copy()
                    selected_idx = tickers.index(whatif_ticker)
                    target_weight = whatif_weight / 100
                    other_total = float(np.sum(np.delete(whatif_weights, selected_idx)))
                    whatif_weights[selected_idx] = target_weight
                    if other_total > 0:
                        scale = (1 - target_weight) / other_total
                        for idx in range(len(tickers)):
                            if idx != selected_idx:
                                whatif_weights[idx] = whatif_weights[idx] * scale
                    else:
                        equal_other = (1 - target_weight) / (len(tickers) - 1)
                        for idx in range(len(tickers)):
                            if idx != selected_idx:
                                whatif_weights[idx] = equal_other
                    split_label = " | ".join(f"{ticker} {weight * 100:.0f}%" for ticker, weight in zip(tickers, whatif_weights))

                wi_ret = portfolio_return(whatif_weights, mean_returns.values)
                wi_var = portfolio_variance(whatif_weights, covariance.values)
                wi_sd = np.sqrt(wi_var)
                wi_esg_score = portfolio_esg(whatif_weights, esg_scores.values)
                wi_sharpe = (wi_ret - risk_free_rate) / wi_sd if wi_sd > 0 else 0.0
                wi_utility = esg_utility_function(wi_ret, wi_var, wi_esg_score, gamma, lambda_esg)

                wi_c1, wi_c2, wi_c3, wi_c4, wi_c5 = st.columns(5)
                render_metric_card(wi_c1, "Split", split_label)
                render_metric_card(wi_c2, "Expected return", f"{wi_ret * 100:.2f}%")
                render_metric_card(wi_c3, "Volatility", f"{wi_sd * 100:.2f}%")
                render_metric_card(wi_c4, "Sharpe ratio", f"{wi_sharpe:.3f}")
                render_metric_card(wi_c5, "ESG score", f"{wi_esg_score:.2f}")

                util_delta = wi_utility - float(recommended["Utility"])
                st.caption(
                    f"Your custom split gives utility {wi_utility:.4f}. "
                    f"The recommended portfolio gives {recommended['Utility']:.4f}. "
                    f"Difference: {util_delta:+.4f}"
                )

                if st.button("Next ", key="next_tab2"):
                    _jump_to_tab(3)

# --------------------------------------------------
# TAB 3 - ESG Breakdown
# --------------------------------------------------
            with tab3:
                render_section_title("ESG Pillar Breakdown")
                render_sage("ESG scores break down into Environmental, Social, and Governance pillars. A high Governance score often predicts lower long-run risk — boards that manage ESG well tend to manage business risk well too.")

                def _pillar_color(v):
                    if v >= 70:
                        return "#34c759"
                    elif v >= 45:
                        return "#f59e0b"
                    return "#ef4444"

                for start in range(0, len(tickers), 2):
                    cols = st.columns(2)
                    for col_idx, ticker in enumerate(tickers[start:start + 2]):
                        esg_data = asset_esg_lookup.get(ticker)
                        with cols[col_idx]:
                            if esg_data is None:
                                st.warning(f"No ESG data for {ticker}")
                            else:
                                _E = min(max(float(esg_data.get("E") or 0), 0), 100)
                                _S = min(max(float(esg_data.get("S") or 0), 0), 100)
                                _G = min(max(float(esg_data.get("G") or 0), 0), 100)
                                _overall = (_E + _S + _G) / 3
                                st.markdown(
                                    f"""<div class="metric-card" style="padding:1.2rem 1.4rem;margin-bottom:0.5rem;">
<div style="font-size:1.1rem;font-weight:700;margin-bottom:0.2rem;">{ticker}</div>
<div style="font-size:2.4rem;font-weight:800;color:{_pillar_color(_overall)};line-height:1.1;">{_overall:.1f} / 100</div>
<div style="font-size:0.75rem;color:#8899bb;margin-bottom:1rem;">Pillar average shown here for breakdown only; optimisation uses your weighted ESG score.</div>""",
                                    unsafe_allow_html=True,
                                )
                                for _label, _val in [("Environmental (E)", _E), ("Social (S)", _S), ("Governance (G)", _G)]:
                                    _color = _pillar_color(_val)
                                    st.markdown(
                                        f'<div style="font-size:0.8rem;color:#8899bb;margin-bottom:0.15rem;">{_label}  <strong style="color:{_color};">{_val:.0f}</strong></div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.progress(_val / 100)

                esg_fig_col1, esg_fig_col2 = st.columns(2)
                with esg_fig_col1:
                    st.pyplot(fig3, width="stretch")
                with esg_fig_col2:
                    st.pyplot(fig4, width="stretch")

                render_section_title("ESG Traffic Lights")
                for start in range(0, len(tickers), 2):
                    cols = st.columns(2)
                    for col_idx, ticker in enumerate(tickers[start:start + 2]):
                        esg_data = asset_esg_lookup.get(ticker)
                        if esg_data is None:
                            cols[col_idx].warning(f"No ESG data for {ticker}")
                            continue
                        E_v = float(esg_data.get("E") or 0)
                        S_v = float(esg_data.get("S") or 0)
                        G_v = float(esg_data.get("G") or 0)

                        def _light(v):
                            if v >= 70:
                                color = "#34c759"
                            elif v >= 45:
                                color = "#f59e0b"
                            else:
                                color = "#ef4444"
                            return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:8px;"></span>'

                        cols[col_idx].markdown(
                            f"""<div class="metric-card" style="padding:1rem 1.2rem;">
<div style="font-size:1rem;font-weight:700;margin-bottom:0.6rem;">{ticker}</div>
<div style="font-size:0.95rem;line-height:2;">
  {_light(E_v)}Environmental (E): <strong>{E_v:.1f}</strong><br>
  {_light(S_v)}Social (S): <strong>{S_v:.1f}</strong><br>
  {_light(G_v)}Governance (G): <strong>{G_v:.1f}</strong>
</div>
<div style="font-size:0.75rem;color:#7b90b8;margin-top:0.5rem;">
  <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#34c759;"></span> &gt;= 70
  &nbsp;&nbsp;
  <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f59e0b;"></span> 45-70
  &nbsp;&nbsp;
  <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;"></span> &lt; 45
</div>
</div>""",
                            unsafe_allow_html=True,
                        )

                # FEATURE 5 — ESG Profile Comparison (grouped bar chart)
                render_section_title("ESG Comparison")
                esg_compare_left, esg_compare_right = st.columns(2, gap="large")
                _esg_comp_tickers = []
                _esg_comp_e = []
                _esg_comp_s = []
                _esg_comp_g = []
                for _t in tickers:
                    _ed = asset_esg_lookup.get(_t)
                    if _ed is not None:
                        _esg_comp_tickers.append(_t)
                        _esg_comp_e.append(float(_ed.get("E") or 0))
                        _esg_comp_s.append(float(_ed.get("S") or 0))
                        _esg_comp_g.append(float(_ed.get("G") or 0))

                if _esg_comp_tickers:
                    _x_pos = np.arange(len(_esg_comp_tickers))
                    _bar_w = 0.23
                    _fig_comp, _ax_comp = plt.subplots(figsize=(5.2, 3.1))
                    _fig_comp.patch.set_facecolor("#f0f7f2")
                    _ax_comp.set_facecolor("#f0f7f2")

                    _ax_comp.bar(_x_pos - _bar_w, _esg_comp_e, _bar_w, label="Environmental (E)", color="#2d8a4e", edgecolor="white", linewidth=0.5)
                    _ax_comp.bar(_x_pos, _esg_comp_s, _bar_w, label="Social (S)", color="#74b99a", edgecolor="white", linewidth=0.5)
                    _ax_comp.bar(_x_pos + _bar_w, _esg_comp_g, _bar_w, label="Governance (G)", color="#c8e6d4", edgecolor="white", linewidth=0.5)

                    _ax_comp.set_xticks(_x_pos)
                    _ax_comp.set_xticklabels(_esg_comp_tickers, fontsize=8.5, fontweight="bold", color="#1a2e1f")
                    _ax_comp.set_ylabel("Score", fontsize=9, color="#5a7a63")
                    _ax_comp.set_title("ESG Profile Comparison", fontsize=10.5, fontweight="bold", color="#1a2e1f")
                    _ax_comp.set_ylim(0, 105)
                    _ax_comp.legend(frameon=False, fontsize=7.8, loc="upper right")
                    _ax_comp.spines["top"].set_visible(False)
                    _ax_comp.spines["right"].set_visible(False)
                    _ax_comp.spines["left"].set_color("#c5d9cc")
                    _ax_comp.spines["bottom"].set_color("#c5d9cc")
                    _ax_comp.tick_params(axis="y", colors="#5a7a63", labelsize=8.5)
                    _ax_comp.grid(axis="y", linestyle="--", alpha=0.3, color="#c5d9cc")
                    plt.tight_layout()
                    with esg_compare_left:
                        st.markdown("##### ESG Profile Comparison")
                        st.pyplot(_fig_comp, width="stretch")
                    plt.close(_fig_comp)
                else:
                    with esg_compare_left:
                        st.markdown("##### ESG Profile Comparison")
                        st.info("No ESG data available for comparison chart.")

                # FEATURE 8 — ESG Peer Comparison
                _benchmarks = {
                    "S&P 500 Avg (Refinitiv 2024)": 54.2,
                    "MSCI World ESG Leaders": 71.3,
                    "Global Market Avg (MSCI 2024)": 49.8,
                    "FTSE4Good Index Avg": 68.5,
                }
                _portfolio_esg_score = float(recommended["ESG_Score"])
                _all_labels = list(_benchmarks.keys()) + ["Your Portfolio"]
                _all_scores = list(_benchmarks.values()) + [_portfolio_esg_score]
                _all_colors = ["#9abfa8"] * len(_benchmarks) + ["#1a6b3c"]

                _fig_peer, _ax_peer = plt.subplots(figsize=(5.2, 3.1))
                _fig_peer.patch.set_facecolor("#f0f7f2")
                _ax_peer.set_facecolor("#f0f7f2")

                _y_pos = np.arange(len(_all_labels))
                _ax_peer.barh(_y_pos, _all_scores, color=_all_colors, height=0.5, edgecolor="white", linewidth=0.5)

                # Add score text at end of each bar
                for _i, (_lbl, _sc) in enumerate(zip(_all_labels, _all_scores)):
                    _ax_peer.text(_sc + 1.0, _i, f"{_sc:.1f}", va="center", fontsize=8.8, fontweight="bold", color="#1a2e1f")

                # Vertical dashed line at portfolio score
                _ax_peer.axvline(x=_portfolio_esg_score, color="#1a6b3c", linestyle="--", linewidth=1.2, alpha=0.7)

                _ax_peer.set_yticks(_y_pos)
                _ax_peer.set_yticklabels(_all_labels, fontsize=8.3, color="#1a2e1f")
                _ax_peer.set_xlim(0, 105)
                _ax_peer.set_xlabel("ESG Score", fontsize=9, color="#5a7a63")
                _ax_peer.set_title("ESG Peer Comparison", fontsize=10.5, fontweight="bold", color="#1a2e1f")
                _ax_peer.spines["top"].set_visible(False)
                _ax_peer.spines["right"].set_visible(False)
                _ax_peer.spines["left"].set_color("#c5d9cc")
                _ax_peer.spines["bottom"].set_color("#c5d9cc")
                _ax_peer.tick_params(axis="x", colors="#5a7a63", labelsize=8.5)
                _ax_peer.grid(axis="x", linestyle="--", alpha=0.3, color="#c5d9cc")
                plt.tight_layout()
                with esg_compare_right:
                    st.markdown("##### ESG Peer Comparison")
                    st.pyplot(_fig_peer, width="stretch")
                plt.close(_fig_peer)
                with esg_compare_right:
                    st.caption(
                        "Benchmark data sourced from MSCI ESG Research and "
                        "Refinitiv ESG scores (2024). Values represent "
                        "weighted average ESG scores for each index."
                    )

                # Verdict: how many benchmarks does the portfolio beat?
                _beat_count = sum(1 for _bv in _benchmarks.values() if _portfolio_esg_score > _bv)
                with esg_compare_right:
                    st.markdown(
                        f'<div style="background:#e8f5ee;border:1px solid #b8d4c2;border-radius:12px;'
                        f'padding:0.7rem 1rem;font-size:0.92rem;color:#1a2e1f;margin-top:0.3rem;">'
                        f'Your portfolio scores higher than <strong>{_beat_count} of {len(_benchmarks)}</strong> '
                        f'benchmarks with an ESG score of <strong>{_portfolio_esg_score:.1f}</strong>.</div>',
                        unsafe_allow_html=True,
                    )

                if st.button("Next ", key="next_tab3"):
                    _jump_to_tab(4)

# --------------------------------------------------
# TAB 5 - Compatibility & Alternatives (merged)
# --------------------------------------------------
            with tab5:
                render_section_title("Recommendation Summary")
                render_recommendation_summary(
                    tickers=tickers,
                    recommended=recommended,
                    tangency=tangency,
                    complete_portfolio=complete_portfolio,
                    market_data=market_data,
                    compatibility_by_asset=compatibility_by_asset,
                    gamma=gamma,
                    lambda_raw_avg=lambda_raw_avg,
                    w_e=w_e,
                    w_s=w_s,
                    w_g=w_g,
                    investment_amount=investment_amount,
                    risk_free_rate=risk_free_rate,
                    profiles=asset_profiles,
                )
                if alternatives_warning:
                    st.info(alternatives_warning)

                render_section_title("Portfolio-Profile Compatibility Score")
                render_sage("Compatibility shows which of your stocks best matches your ESG preference profile — not just overall ESG score, but how well each pillar aligns with what you said matters most to you.")

                overall = compat["overall_compatibility"]
                risk_fit = compat["risk_compatibility"]
                esg_fit = compat["esg_compatibility"]
                pillar_fit = compat["pillar_alignment"]

                if overall >= 80:
                    verdict_text = "Excellent fit  this portfolio closely reflects your risk and sustainability values."
                    verdict_color = "#166534"
                    verdict_bg = "rgba(52,199,89,0.10)"
                elif overall >= 60:
                    verdict_text = "Good fit  minor trade-offs between your ESG priorities and financial goals."
                    verdict_color = "#92400e"
                    verdict_bg = "rgba(245,158,11,0.10)"
                elif overall >= 40:
                    verdict_text = "Partial fit  consider adjusting your asset choices or preferences."
                    verdict_color = "#b45309"
                    verdict_bg = "rgba(245,158,11,0.07)"
                else:
                    verdict_text = "Low fit  the selected stocks may not align well with your stated values."
                    verdict_color = "#991b1b"
                    verdict_bg = "rgba(239,68,68,0.10)"

                st.markdown(
                    f"""<div style="display:flex;flex-direction:column;align-items:center;
padding:2rem;background:{verdict_bg};border-radius:24px;margin-bottom:1.4rem;
border:1px solid rgba(15,23,42,0.08);">
  <div style="font-size:5rem;font-weight:800;color:{verdict_color};line-height:1;">
    {int(overall)}%
  </div>
  <div style="font-size:1rem;font-weight:600;color:{verdict_color};margin-top:0.4rem;">
    Overall Compatibility
  </div>
  <div style="font-size:0.95rem;color:#4a7a58;margin-top:0.8rem;text-align:center;max-width:540px;">
    {verdict_text}
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

                def _progress_bar(label, value, color):
                    pct = min(max(int(value), 0), 100)
                    st.markdown(
                        f"""<div style="margin-bottom:1rem;">
<div style="display:flex;justify-content:space-between;font-size:0.88rem;
color:#4a7a58;margin-bottom:0.3rem;">
  <span><strong>{label}</strong></span>
  <span style="color:{color};font-weight:700;">{pct}%</span>
</div>
<div style="background:#d4e8dc;border-radius:999px;height:10px;overflow:hidden;">
  <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;
  transition:width 0.4s ease;"></div>
</div>
</div>""",
                        unsafe_allow_html=True,
                    )

                _progress_bar("Risk Fit", risk_fit, "#0071e3")
                _progress_bar("ESG Fit", esg_fit, "#34c759")
                _progress_bar("Pillar Alignment", pillar_fit, "#5e5ce6")

                # ---- Integrated: ESG Investor Profile & Alignment Cards ----
                render_section_title("ESG Investor Profile & Portfolio Match")
                st.markdown(
                    f"""<div class="hero-card" style="padding:1.2rem 1.6rem; margin-bottom:1rem;">
<div style="font-size:0.82rem;font-weight:600;color:#0071e3;margin-bottom:0.6rem;letter-spacing:0.04em;">YOUR ESG INVESTOR PROFILE</div>
<div style="display:flex;gap:2.5rem;flex-wrap:wrap;">
  <div><div style="font-size:0.78rem;color:#7b90b8;margin-bottom:0.15rem;">Primary ESG Focus</div><strong style="font-size:1rem;color:{focus_color};">{primary_focus}</strong></div>
  <div><div style="font-size:0.78rem;color:#7b90b8;margin-bottom:0.15rem;">ESG Intensity</div><strong style="font-size:1rem;">{classify_esg(lambda_raw_avg)}</strong></div>
  <div><div style="font-size:0.78rem;color:#7b90b8;margin-bottom:0.15rem;">Risk Profile</div><strong style="font-size:1rem;">{classify_risk(gamma)}</strong></div>
  <div><div style="font-size:0.78rem;color:#7b90b8;margin-bottom:0.15rem;">E / S / G Weights</div><strong style="font-size:1rem;">{w_e:.0%} / {w_s:.0%} / {w_g:.0%}</strong></div>
</div>
</div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("**Stock-to-Profile Alignment**")
                st.caption(
                    "Composite score = 60% weighted ESG quality + 40% profile match."
                )
                def render_alignment_card(col, ticker_label, composite, profile_match, quality, gaps, esg_data):
                    badge_label, badge_color = alignment_badge(composite)
                    if composite is None:
                        col.markdown(
                            f"""<div class="metric-card"><div class="metric-card-label">{ticker_label}</div>
<div style="color:#7b90b8;font-size:0.9rem;margin-top:0.4rem;">ESG data unavailable</div></div>""",
                            unsafe_allow_html=True,
                        )
                        return
                    E_raw = float(esg_data.get("E") or 0)
                    S_raw = float(esg_data.get("S") or 0)
                    G_raw = float(esg_data.get("G") or 0)
                    e_gap, s_gap, g_gap = gaps
                    worst = max([("E", e_gap), ("S", s_gap), ("G", g_gap)], key=lambda x: x[1])
                    gap_html = (
                        f'<div style="font-size:0.82rem;color:#92400e;margin-top:0.55rem;padding:0.35rem 0.6rem;'
                        f'background:rgba(245,158,11,0.08);border-radius:8px;">'
                        f'Underdelivers on: <strong>{worst[0]}</strong> (gap {worst[1]:+.2f})</div>'
                        if worst[1] > 0.05
                        else '<div style="font-size:0.82rem;color:#34c759;margin-top:0.55rem;padding:0.35rem 0.6rem;'
                             'background:rgba(52,199,89,0.08);border-radius:8px;">Delivers well across all your priorities</div>'
                    )

                    def _pbar(label, value, color, investor_w):
                        pct = min(max(float(value), 0), 100)
                        inv_pct = min(max(float(investor_w) * 100, 0), 100)
                        return (
                            f'<div style="margin-bottom:0.5rem;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:0.76rem;'
                            f'color:#4b5563;margin-bottom:0.18rem;">'
                            f'<span>{label}</span>'
                            f'<span><strong style="color:#111827;">{pct:.0f}</strong>'
                            f'<span style="color:#334155;"> / you: {inv_pct:.0f}</span></span></div>'
                            f'<div style="position:relative;background:#d8e1ea;border-radius:999px;height:8px;overflow:hidden;">'
                            f'<div style="width:{pct}%;height:100%;background:{color};border-radius:999px;"></div>'
                            f'<div style="position:absolute;top:0;left:{inv_pct}%;width:2px;height:100%;'
                            f'background:#1f2933;opacity:0.45;"></div>'
                            f'</div></div>'
                        )

                    bars_html = (
                        _pbar("Environmental (E)", E_raw, "#2f7a59", w_e) +
                        _pbar("Social (S)", S_raw, "#2d5c88", w_s) +
                        _pbar("Governance (G)", G_raw, "#5a6f8c", w_g)
                    )
                    ring_pct = min(int(composite), 100)
                    col.markdown(
                        f"""<div class="metric-card" style="padding:1.15rem 1.25rem;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.8rem;">
  <div>
    <div style="font-size:0.95rem;font-weight:700;color:#1f2933;margin-bottom:0.3rem;">{ticker_label}</div>
    <span style="padding:0.2rem 0.6rem;border-radius:999px;background:{badge_color}22;color:{badge_color};font-size:0.74rem;font-weight:600;">{badge_label}</span>
  </div>
  <div style="text-align:center;min-width:58px;">
    <div style="font-size:2.2rem;font-weight:700;color:{badge_color};line-height:1;">{ring_pct}</div>
    <div style="font-size:0.69rem;color:#4b5563;font-weight:500;">/100 match</div>
  </div>
</div>
<div style="font-size:0.78rem;color:#334155;margin-bottom:0.7rem;">
  Profile match: <strong style="color:#111827;">{profile_match:.1f}%</strong>
  &nbsp;&nbsp; ESG quality: <strong style="color:#111827;">{quality:.1f}%</strong>
</div>
<div style="font-size:0.72rem;color:#64748b;margin-bottom:0.5rem;font-style:italic;">Bar = stock score &nbsp;|&nbsp; tick = your priority weight</div>
{bars_html}
{gap_html}
</div>""",
                        unsafe_allow_html=True,
                    )

                for start in range(0, len(tickers), 2):
                    cols = st.columns(2)
                    for col_idx, ticker in enumerate(tickers[start:start + 2]):
                        composite, profile_match, quality, _, _, _, gaps = asset_alignment[ticker]
                        render_alignment_card(cols[col_idx], ticker, composite, profile_match, quality, gaps, asset_esg_lookup.get(ticker))

                # ---- Bridge: explain why alternatives are shown ----
                st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

                # Determine if alternatives are worth showing based on score
                _show_alt_reason = ""
                if overall < 80:
                    if overall >= 60:
                        _show_alt_reason = f"Your portfolio scores <strong>{int(overall)}%</strong> compatibility. The stocks below are better aligned to your ESG profile and could push your score higher."
                    else:
                        _show_alt_reason = f"Your portfolio scores <strong>{int(overall)}%</strong> compatibility — there's meaningful room to improve. Swapping in one of these alternatives could significantly raise your score."
                else:
                    _show_alt_reason = f"Your portfolio already scores <strong>{int(overall)}%</strong>. These alternatives are shown for reference — they match your ESG profile even more closely."

                st.markdown(
                    f"""<div style="background:linear-gradient(135deg,#f0f7f2,#e8f5ee);
                    border:1px solid #b8d4c2;border-left:4px solid #1a6b3c;
                    border-radius:16px;padding:1.2rem 1.6rem;margin-bottom:1.4rem;">
                    <div style="font-size:0.75rem;font-weight:700;color:#1a6b3c;
                    letter-spacing:0.07em;margin-bottom:0.4rem;">IMPROVE YOUR COMPATIBILITY SCORE</div>
                    <p style="font-size:0.96rem;color:#1a2e1f;margin:0;line-height:1.7;">
                    {_show_alt_reason}
                    </p></div>""",
                    unsafe_allow_html=True,
                )

                render_section_title("Suggested Alternatives")
                st.caption("Each alternative is scored against your ESG profile. Higher alignment = better compatibility fit.")
                if not candidates.empty:
                    rec_uni = candidates.dropna(subset=["E", "S", "G", "Expected_Return"]).copy()
                    if not rec_uni.empty:
                        rec_uni = rec_uni[~rec_uni["ticker"].isin(tickers)]
                        tot = rec_uni["E"] + rec_uni["S"] + rec_uni["G"]
                        tot = tot.replace(0, np.nan)
                        se_col = (rec_uni["E"] / tot).fillna(1 / 3)
                        ss_col = (rec_uni["S"] / tot).fillna(1 / 3)
                        sg_col = (rec_uni["G"] / tot).fillna(1 / 3)
                        dot_col = w_e * se_col + w_s * ss_col + w_g * sg_col
                        mag_inv_val = float(np.sqrt(w_e ** 2 + w_s ** 2 + w_g ** 2)) or 1e-9
                        mag_stock_col = np.sqrt(se_col ** 2 + ss_col ** 2 + sg_col ** 2).replace(0, 1e-9)
                        rec_uni["_profile_match"] = (dot_col / (mag_inv_val * mag_stock_col)) * 100
                        rec_uni["_quality"] = (w_e * rec_uni["E"] + w_s * rec_uni["S"] + w_g * rec_uni["G"])
                        rec_uni["_composite"] = 0.6 * rec_uni["_quality"] + 0.4 * rec_uni["_profile_match"]
                        ret_threshold = float(mean_returns.min()) * 0.9
                        rec_filtered = rec_uni[rec_uni["Expected_Return"] >= ret_threshold].sort_values("_composite", ascending=False)
                        top3 = rec_filtered.head(3)
                        weakest_asset = min((item for item in compatibility_by_asset if item["composite"] is not None), key=lambda x: x["composite"], default=None)

                        if top3.empty:
                            st.info("No profile-matched alternatives found in the ESG universe with sufficient return data.")
                        else:
                            dim_full = {"E": "Environmental", "S": "Social", "G": "Governance"}
                            for _, row in top3.iterrows():
                                cs = float(row["_composite"])
                                pm_val = float(row["_profile_match"])
                                q_val = float(row["_quality"])
                                badge_label_alt, badge_color_alt = alignment_badge(cs)
                                E_r = float(row.get("E") or 0)
                                S_r = float(row.get("S") or 0)
                                G_r = float(row.get("G") or 0)
                                exp_ret = float(row.get("Expected_Return") or 0)
                                vol_val = float(row.get("Volatility") or 0)
                                best_dim = max(
                                    {"E": E_r * w_e, "S": S_r * w_s, "G": G_r * w_g},
                                    key=lambda k: {"E": E_r * w_e, "S": S_r * w_s, "G": G_r * w_g}[k],
                                )
                                why = (
                                    f"Strong {dim_full[best_dim]} score ({best_dim}: {row.get(best_dim, 0):.3f}), "
                                    f"matching your {dim_full[best_dim]} priority ({dim_weights[best_dim]:.0%} weight)."
                                )
                                vs_note = ""
                                if weakest_asset and cs > weakest_asset["composite"]:
                                    vs_note = f" Alignment +{cs - weakest_asset['composite']:.1f}pts vs {weakest_asset['ticker']}."
                                st.markdown(
                                    f"""<div class="green-card" style="margin-bottom:0.75rem;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;">
  <div>
    <div style="font-size:1.08rem;font-weight:700;color:#2f7a59;">{row['ticker']}</div>
    <div style="font-size:0.83rem;color:#667380;margin-top:0.2rem;">{why}{vs_note}</div>
  </div>
  <div style="text-align:right;">
    <span style="display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;background:{badge_color_alt}22;color:{badge_color_alt};font-size:0.8rem;font-weight:600;">{badge_label_alt}</span>
    <div style="font-size:0.92rem;color:#2f7a59;font-weight:700;margin-top:0.2rem;">Alignment: {cs:.1f}/100</div>
    <div style="font-size:0.78rem;color:#4a7a58;margin-top:0.15rem;">{'↑ Higher than your portfolio avg' if weakest_asset and cs > weakest_asset['composite'] else 'Solid ESG match'}</div>
  </div>
</div>
<div style="display:flex;gap:1.4rem;flex-wrap:wrap;margin-top:0.55rem;font-size:0.84rem;color:#355a49;">
  <div>Return: <strong>{exp_ret:.2%}</strong></div>
  <div>Volatility: <strong>{vol_val:.2%}</strong></div>
  <div>E: <strong>{E_r:.3f}</strong> | S: <strong>{S_r:.3f}</strong> | G: <strong>{G_r:.3f}</strong></div>
  <div>Profile Match: <strong>{pm_val:.1f}%</strong> | ESG Quality: <strong>{q_val:.1f}%</strong></div>
</div>
</div>""",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("No alternative stocks with sufficient data found.")
                else:
                    st.info("ESG universe data is not available for profile-matched recommendations.")

                st.markdown('<hr style="margin:2rem 0;">', unsafe_allow_html=True)

                # Download report
                report_lines = [
                    "SUSTAINABLE PORTFOLIO RECOMMENDER  INVESTMENT SUMMARY",
                    "=" * 56,
                    "",
                    "INVESTOR PROFILE",
                    f"  Risk attitude      : {classify_risk(gamma)}",
                    f"  Sustainability     : {classify_esg(lambda_raw_avg)}",
                    f"  Risk aversion (gamma)  : {gamma:.2f}",
                    f"  ESG preference (lambda): {lambda_raw_avg:.2f} / 4",
                    f"  E / S / G weights  : {w_e:.0%} / {w_s:.0%} / {w_g:.0%}",
                    "",
                    "RECOMMENDED PORTFOLIO",
                ]
                for ticker, weight in zip(tickers, recommended_weights):
                    report_lines.append(f"  {ticker:<16}: {weight * 100:.2f}%")
                report_lines.extend([
                    f"  Expected return    : {recommended['Expected_Return']*100:.2f}%",
                    f"  Volatility         : {recommended['Risk_SD']*100:.2f}%",
                    f"  Sharpe ratio       : {recommended['Sharpe_Ratio']:.3f}",
                    f"  ESG score          : {recommended['ESG_Score']:.2f}",
                    "",
                    "COMPATIBILITY SCORES",
                    f"  Overall            : {overall:.1f}%",
                    f"  Risk Fit           : {risk_fit:.1f}%",
                    f"  ESG Fit            : {esg_fit:.1f}%",
                    f"  Pillar Alignment   : {pillar_fit:.1f}%",
                    "",
                    "VERDICT",
                    f"  {verdict_text}",
                    "",
                    "Generated by Sustainable Portfolio Recommender",
                ])
                report_text = "\n".join(report_lines)

                st.download_button(
                    label="Download My Report",
                    data=report_text,
                    file_name="portfolio_report.txt",
                    mime="text/plain",
                )

        except Exception as e:
            with tab1:
                st.error(str(e))
elif profile_complete:
    with tab1:
        st.info("Set your inputs in the sidebar, then click 'Run portfolio optimisation'.")














