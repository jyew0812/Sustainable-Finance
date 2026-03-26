import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import yfinance as yf
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Sustainable Portfolio Recommender", layout="wide")


def inject_apple_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {
                --bg: #f5f5f7;
                --surface: rgba(255, 255, 255, 0.88);
                --surface-strong: #ffffff;
                --border: rgba(15, 23, 42, 0.08);
                --text: #111827;
                --muted: #6b7280;
                --accent: #0071e3;
                --accent-soft: #e8f2ff;
                --shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
                --radius: 24px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(0, 113, 227, 0.08), transparent 28%),
                    radial-gradient(circle at top right, rgba(52, 199, 89, 0.08), transparent 24%),
                    linear-gradient(180deg, #fbfbfd 0%, #f2f4f7 100%);
                color: var(--text);
                font-family: 'Inter', sans-serif;
            }

            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0.72);
                backdrop-filter: blur(14px);
            }

            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.78);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] > div:first-child {
                background: transparent;
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, label, p, span, div {
                color: var(--text);
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.82));
                border: 1px solid var(--border);
                border-radius: 30px;
                padding: 2rem 2.2rem;
                box-shadow: var(--shadow);
                margin-bottom: 1.4rem;
            }

            .hero-eyebrow {
                display: inline-block;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: var(--accent-soft);
                color: var(--accent);
                font-size: 0.82rem;
                font-weight: 600;
                margin-bottom: 0.9rem;
            }

            .hero-title {
                font-size: clamp(2rem, 4vw, 3.4rem);
                line-height: 1.02;
                font-weight: 700;
                letter-spacing: -0.04em;
                margin: 0;
            }

            .hero-subtitle {
                margin-top: 0.9rem;
                max-width: 720px;
                font-size: 1.02rem;
                line-height: 1.7;
                color: var(--muted);
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 0.9rem 1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
                min-height: 150px;
            }

            [data-testid="stMetricLabel"] {
                font-size: 0.84rem !important;
                line-height: 1.35 !important;
                white-space: normal !important;
                color: #4b5563 !important;
            }

            [data-testid="stMetricValue"] {
                font-size: clamp(1.35rem, 2vw, 2.55rem) !important;
                line-height: 1.12 !important;
                white-space: normal !important;
                word-break: break-word !important;
                overflow-wrap: anywhere !important;
            }

            .metric-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 22px;
                padding: 0.95rem 1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
                min-height: 150px;
            }

            .metric-card-label {
                font-size: 0.88rem;
                line-height: 1.35;
                color: #4b5563;
                margin-bottom: 0.55rem;
                word-break: break-word;
                overflow-wrap: anywhere;
            }

            .metric-card-value {
                font-size: clamp(1.25rem, 1.7vw, 2.25rem);
                line-height: 1.1;
                color: #111827;
                font-weight: 600;
                letter-spacing: -0.03em;
                word-break: break-word;
                overflow-wrap: anywhere;
            }

            .green-card {
                background: linear-gradient(135deg, rgba(52, 199, 89, 0.18), rgba(52, 199, 89, 0.08));
                border: 1px solid rgba(52, 199, 89, 0.28);
                border-radius: 22px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
                min-height: 150px;
            }

            .green-card-title {
                font-size: 0.9rem;
                line-height: 1.35;
                color: #166534;
                font-weight: 700;
                margin-bottom: 0.65rem;
                letter-spacing: 0.01em;
            }

            .green-card-line {
                font-size: 0.98rem;
                line-height: 1.55;
                color: #14532d;
            }

            .warning-text {
                color: #dc2626;
                font-size: 0.92rem;
                font-weight: 700;
                line-height: 1.4;
                margin-top: 0.2rem;
            }

            [data-testid="stDataFrame"],
            [data-testid="stPlotlyChart"],
            [data-testid="stImage"],
            [data-testid="stTable"],
            .stAlert {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            }

            .stAlert {
                padding: 0.9rem 1rem;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            .stNumberInput > div > div,
            .stTextInput > div > div {
                border-radius: 14px !important;
                border-color: rgba(15, 23, 42, 0.08) !important;
                background: rgba(255, 255, 255, 0.96) !important;
            }

            div[data-baseweb="select"] > div {
                color: #111827 !important;
            }

            div[data-baseweb="popover"] *,
            ul[role="listbox"] *,
            li[role="option"] * {
                background: #ffffff !important;
                color: #111827 !important;
            }

            li[role="option"] {
                background: #ffffff !important;
                color: #111827 !important;
            }

            li[role="option"]:hover,
            li[role="option"][aria-selected="true"] {
                background: #f3f4f6 !important;
                color: #111827 !important;
            }

            div[data-baseweb="input"] input,
            div[data-baseweb="select"] input,
            .stTextInput input,
            .stNumberInput input,
            textarea {
                color: #111827 !important;
                -webkit-text-fill-color: #111827 !important;
                caret-color: #111827 !important;
            }

            div[data-baseweb="input"] input::placeholder,
            .stTextInput input::placeholder,
            .stNumberInput input::placeholder,
            textarea::placeholder {
                color: #9ca3af !important;
                -webkit-text-fill-color: #9ca3af !important;
            }

            div[data-baseweb="select"] * ,
            .stSelectbox * ,
            .stTextInput label,
            .stNumberInput label,
            .stSelectbox label,
            .stSlider label {
                color: #111827 !important;
            }

            .stRadio > div {
                gap: 0.35rem;
            }

            .stRadio label {
                background: rgba(255,255,255,0.74);
                border: 1px solid rgba(15, 23, 42, 0.06);
                padding: 0.55rem 0.75rem;
                border-radius: 14px;
            }

            .stButton > button {
                background: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(15, 23, 42, 0.1) !important;
                border-radius: 999px;
                padding: 0.72rem 1.15rem;
                font-weight: 600;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            }

            .stButton > button:hover {
                background: #f8fafc !important;
                color: #111827 !important;
            }

            .section-title {
                font-size: 1.35rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                margin-top: 0.25rem;
                margin-bottom: 0.8rem;
            }

            [data-testid="stDataFrame"] *,
            [data-testid="stTable"] *,
            [data-testid="stDataFrameResizable"] *,
            [data-testid="stTable"] table,
            [data-testid="stTable"] th,
            [data-testid="stTable"] td {
                color: #111827 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Sustainable Investing</div>
            <h1 class="hero-title">Sustainable Portfolio Recommender</h1>
            <p class="hero-subtitle">
                Build a polished two-asset portfolio recommendation using market data, personal risk preferences,
                and your own ESG priorities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_metric_card(column, label, value):
    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_green_cost_card(sharpe_gap, esg_gain):
    st.markdown(
        f"""
        <div class="green-card">
            <div class="green-card-title">Green Cost Calculator</div>
            <div class="green-card-line"><strong>Sharpe ratio difference:</strong> {sharpe_gap:+.3f}</div>
            <div class="green-card-line"><strong>ESG score gain:</strong> {esg_gain:+.2f} points</div>
            <div class="green-card-line"><strong>Formula:</strong> Tangency Sharpe ratio - Recommended Sharpe ratio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sin_warning():
    st.sidebar.markdown('<div class="warning-text">This is not green.</div>', unsafe_allow_html=True)


def style_table(df):
    return (
        df.style
        .hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("width", "100%"),
                        ("border-collapse", "separate"),
                        ("border-spacing", "0"),
                        ("background", "#ffffff"),
                        ("color", "#111827"),
                        ("border", "1px solid rgba(15, 23, 42, 0.08)"),
                        ("border-radius", "18px"),
                        ("overflow", "hidden"),
                        ("box-shadow", "0 10px 30px rgba(15, 23, 42, 0.05)"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background", "#f8fafc"),
                        ("color", "#111827"),
                        ("font-weight", "600"),
                        ("text-align", "left"),
                        ("padding", "12px 14px"),
                        ("border-bottom", "1px solid rgba(15, 23, 42, 0.08)"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background", "#ffffff"),
                        ("color", "#111827"),
                        ("padding", "12px 14px"),
                        ("border-bottom", "1px solid rgba(15, 23, 42, 0.06)"),
                    ],
                },
            ]
        )
    )


inject_apple_theme()
render_hero()


# =========================
# Core functions
# =========================
def classify_risk(gamma):
    if gamma >= 8:
        return "Defensive"
    elif gamma >= 5:
        return "Balanced"
    return "Growth-Oriented"


def classify_esg(lambda_raw_avg):
    if lambda_raw_avg >= 7:
        return "Sustainability-Led"
    elif lambda_raw_avg >= 4:
        return "ESG-Aware"
    return "Low ESG Priority"


SIN_INDUSTRIES = [
    "Tobacco",
    "Gambling",
    "Resorts & Casinos",
    "Oil & Gas E&P",
    "Oil & Gas Integrated",
    "Oil & Gas Midstream",
    "Oil & Gas Refining & Marketing",
    "Oil & Gas Equipment & Services",
    "Oil & Gas Drilling",
    "Thermal Coal",
]


def is_sin_industry(industry):
    return industry in SIN_INDUSTRIES


@st.cache_data(show_spinner=False)
def fetch_ticker_profile(ticker):
    if not ticker:
        return None

    ticker_obj = yf.Ticker(ticker)
    info = {}

    try:
        info = ticker_obj.get_info() or {}
    except Exception:
        info = {}

    if not info:
        try:
            info = ticker_obj.info or {}
        except Exception:
            info = {}

    sector = (
        info.get("sector")
        or info.get("sectorDisp")
        or info.get("sectorKey")
        or "Unavailable"
    )
    industry = (
        info.get("industry")
        or info.get("industryDisp")
        or info.get("industryKey")
        or "Unavailable"
    )
    short_name = (
        info.get("shortName")
        or info.get("longName")
        or info.get("displayName")
        or ticker
    )

    if sector == "Unavailable" or industry == "Unavailable":
        try:
            search = yf.Search(ticker, max_results=1)
            quotes = getattr(search, "quotes", []) or []
        except Exception:
            quotes = []

        if quotes:
            quote = quotes[0]
            short_name = (
                quote.get("shortname")
                or quote.get("longname")
                or quote.get("dispSecIndFlag")
                or short_name
            )
            sector = (
                quote.get("sector")
                or quote.get("sectorDisp")
                or quote.get("sectorKey")
                or sector
            )
            industry = (
                quote.get("industry")
                or quote.get("industryDisp")
                or quote.get("industryKey")
                or industry
            )

    if sector == "Unavailable" or industry == "Unavailable":
        try:
            quote_type = ticker_obj.get_history_metadata() or {}
        except Exception:
            quote_type = {}

        short_name = quote_type.get("shortName") or short_name

    return {
        "name": short_name,
        "sector": sector,
        "industry": industry,
    }


@st.cache_data(show_spinner=False)
def fetch_market_data(ticker1, ticker2, period):
    data = yf.download([ticker1, ticker2], period=period, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError("No data was downloaded. Check the tickers and try again.")

    if "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        raise ValueError("Could not retrieve both tickers properly.")

    prices = prices[[ticker1, ticker2]].dropna()

    if prices.empty or len(prices) < 30:
        raise ValueError("Insufficient price history. Choose different tickers or a longer period.")

    returns = prices.pct_change().dropna()

    mean_returns = returns.mean() * 252
    volatilities = returns.std() * np.sqrt(252)
    correlation = returns[ticker1].corr(returns[ticker2])
    return {
        "prices": prices,
        "returns": returns,
        "r1": float(mean_returns[ticker1]),
        "r2": float(mean_returns[ticker2]),
        "sd1": float(volatilities[ticker1]),
        "sd2": float(volatilities[ticker2]),
        "corr": float(correlation),
    }


def portfolio_return(w1, r1, r2):
    w2 = 1 - w1
    return w1 * r1 + w2 * r2


def portfolio_variance(w1, sd1, sd2, corr):
    w2 = 1 - w1
    return (w1 ** 2) * (sd1 ** 2) + (w2 ** 2) * (sd2 ** 2) + 2 * w1 * w2 * corr * sd1 * sd2


def portfolio_esg(w1, esg1, esg2):
    w2 = 1 - w1
    return w1 * esg1 + w2 * esg2


def esg_utility_function(port_return_value, port_variance, port_esg_score, gamma, lambda_esg):
    return port_return_value - 0.5 * gamma * port_variance + lambda_esg * port_esg_score


def utility_function(expected_complete_return, complete_variance, gamma):
    return expected_complete_return - 0.5 * gamma * complete_variance


def build_portfolio_table(r1, r2, sd1, sd2, corr, rf, esg1, esg2, gamma, lambda_esg):
    weights = np.linspace(0, 1, 1001)
    rows = []

    for w1 in weights:
        w2 = 1 - w1
        ret = portfolio_return(w1, r1, r2)
        var = portfolio_variance(w1, sd1, sd2, corr)
        risk = np.sqrt(var)
        esg_score = portfolio_esg(w1, esg1, esg2)
        sharpe = (ret - rf) / risk if risk > 0 else np.nan
        utility = esg_utility_function(ret, var, esg_score, gamma, lambda_esg)

        rows.append({
            "Weight_Asset1": w1,
            "Weight_Asset2": w2,
            "Expected_Return": ret,
            "Variance": var,
            "Risk_SD": risk,
            "ESG_Score": esg_score,
            "Sharpe_Ratio": sharpe,
            "Utility": utility,
        })

    return pd.DataFrame(rows)


def select_recommended_portfolio(df):
    return df.loc[df["Utility"].idxmax()]


def select_max_sharpe_portfolio(df):
    return df.loc[df["Sharpe_Ratio"].idxmax()]


def build_complete_portfolio(tangency_portfolio, rf, gamma):
    tangency_excess_return = tangency_portfolio["Expected_Return"] - rf
    tangency_variance = tangency_portfolio["Variance"]

    if tangency_variance <= 0:
        raise ValueError("Tangency portfolio variance must be positive to compute the complete portfolio.")

    y = tangency_excess_return / (gamma * tangency_variance)
    expected_complete_return = rf + y * tangency_excess_return
    complete_variance = (y ** 2) * tangency_variance
    complete_risk = np.sqrt(complete_variance)
    utility = utility_function(expected_complete_return, complete_variance, gamma)

    return {
        "y": y,
        "weight_risk_free": 1 - y,
        "Expected_Return": expected_complete_return,
        "Variance": complete_variance,
        "Risk_SD": complete_risk,
        "Utility": utility,
        "Weight_Asset1": y * tangency_portfolio["Weight_Asset1"],
        "Weight_Asset2": y * tangency_portfolio["Weight_Asset2"],
        "ESG_Score": tangency_portfolio["ESG_Score"],
        "Sharpe_Ratio": tangency_portfolio["Sharpe_Ratio"],
    }


def style_axis(ax, x_percent=False, y_percent=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#374151", labelsize=11)
    ax.grid(True, color="#e5e7eb", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.title.set_color("#111827")
    ax.xaxis.label.set_color("#374151")
    ax.yaxis.label.set_color("#374151")

    if x_percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    if y_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))


def add_end_label(ax, x_value, y_value, label, color):
    ax.annotate(
        label,
        (x_value, y_value),
        textcoords="offset points",
        xytext=(8, 0),
        va="center",
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#e5e7eb", alpha=0.95)
    )


def make_frontier_figure(df, tangency, recommended, ticker1, ticker2, r1, r2, sd1, sd2):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    frontier_color = "#0071e3"
    tangency_color = "#111827"
    recommended_color = "#34c759"

    ax.plot(df["Risk_SD"], df["Expected_Return"], linewidth=2.8, color=frontier_color)
    ax.scatter(
        tangency["Risk_SD"], tangency["Expected_Return"], s=150, marker="X",
        color=tangency_color, linewidths=1.2
    )
    ax.scatter(
        recommended["Risk_SD"], recommended["Expected_Return"], s=150, marker="D",
        color=recommended_color, edgecolors="white", linewidths=1.2
    )
    ax.scatter([sd1, sd2], [r1, r2], s=105, color="#a1a1aa")

    add_end_label(ax, df["Risk_SD"].iloc[-1], df["Expected_Return"].iloc[-1], "Frontier", frontier_color)
    add_end_label(ax, tangency["Risk_SD"], tangency["Expected_Return"], "Tangency portfolio", tangency_color)
    add_end_label(ax, recommended["Risk_SD"], recommended["Expected_Return"], "Recommended portfolio", recommended_color)
    ax.annotate(ticker1, (sd1, r1), textcoords="offset points", xytext=(6, 6), fontsize=10)
    ax.annotate(ticker2, (sd2, r2), textcoords="offset points", xytext=(6, 6), fontsize=10)

    ax.set_xlabel("Portfolio Risk (Standard Deviation)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Risky Asset Frontier")
    style_axis(ax, x_percent=True, y_percent=True)
    fig.tight_layout()
    return fig


def make_cml_figure(df, tangency, complete_portfolio, rf, ticker1, ticker2, r1, r2, sd1, sd2):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    frontier_color = "#0071e3"
    cml_color = "#34c759"
    tangency_color = "#111827"
    complete_color = "#ff9f0a"
    risk_free_color = "#5e5ce6"

    cml_risk = np.linspace(0, max(df["Risk_SD"].max(), complete_portfolio["Risk_SD"]) * 1.15, 300)
    cml_return = rf + tangency["Sharpe_Ratio"] * cml_risk

    ax.plot(df["Risk_SD"], df["Expected_Return"], linewidth=2.8, color=frontier_color)
    ax.plot(cml_risk, cml_return, linewidth=2.8, color=cml_color, linestyle="--")
    ax.scatter(
        tangency["Risk_SD"], tangency["Expected_Return"], s=160, marker="X",
        color=tangency_color, linewidths=1.2
    )
    ax.scatter(
        complete_portfolio["Risk_SD"], complete_portfolio["Expected_Return"], s=150, marker="D",
        color=complete_color, edgecolors="white", linewidths=1.2
    )
    ax.scatter(0, rf, s=110, color=risk_free_color)
    ax.scatter([sd1, sd2], [r1, r2], s=95, color="#a1a1aa")

    add_end_label(ax, df["Risk_SD"].iloc[-1], df["Expected_Return"].iloc[-1], "Frontier", frontier_color)
    add_end_label(ax, cml_risk[-1], cml_return[-1], "Capital market line", cml_color)
    add_end_label(ax, tangency["Risk_SD"], tangency["Expected_Return"], "Tangency portfolio", tangency_color)
    add_end_label(
        ax,
        complete_portfolio["Risk_SD"],
        complete_portfolio["Expected_Return"],
        "Optimal complete portfolio",
        complete_color,
    )
    ax.annotate("Risk-free", (0, rf), textcoords="offset points", xytext=(8, 8), fontsize=10)
    ax.annotate(ticker1, (sd1, r1), textcoords="offset points", xytext=(6, 6), fontsize=10)
    ax.annotate(ticker2, (sd2, r2), textcoords="offset points", xytext=(6, 6), fontsize=10)

    ax.set_xlabel("Portfolio Risk (Standard Deviation)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Efficient Frontier and Capital Market Line")
    style_axis(ax, x_percent=True, y_percent=True)
    fig.tight_layout()
    return fig


def make_esg_tradeoff_figure(df, recommended):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    tradeoff_color = "#5e5ce6"
    recommended_color = "#34c759"

    ax.plot(df["ESG_Score"], df["Expected_Return"], linewidth=2.8, color=tradeoff_color)
    ax.scatter(
        recommended["ESG_Score"], recommended["Expected_Return"], s=150, marker="D",
        color=recommended_color, edgecolors="white", linewidths=1.2
    )

    add_end_label(ax, df["ESG_Score"].iloc[-1], df["Expected_Return"].iloc[-1], "Trade-off", tradeoff_color)
    add_end_label(ax, recommended["ESG_Score"], recommended["Expected_Return"], "Recommended portfolio", recommended_color)

    ax.set_xlabel("Portfolio ESG Score")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("ESG Portfolio Frontier and Recommended Portfolio")
    style_axis(ax, y_percent=True)
    fig.tight_layout()
    return fig


def make_esg_efficient_frontier_figure(df, tangency, recommended):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    curve = df.sort_values("ESG_Score")
    frontier_color = "#5e5ce6"
    tangency_color = "#ff6b6b"
    recommended_color = "#34c759"
    guide_color = "#f59e0b"

    ax.plot(curve["ESG_Score"], curve["Sharpe_Ratio"], linewidth=2.8, color=frontier_color)
    ax.scatter(
        tangency["ESG_Score"], tangency["Sharpe_Ratio"], s=220, marker="*",
        color=tangency_color, edgecolors="white", linewidths=1.0
    )
    ax.scatter(
        recommended["ESG_Score"], recommended["Sharpe_Ratio"], s=180, marker="D",
        color=recommended_color, edgecolors="white", linewidths=1.2
    )

    ax.hlines(
        tangency["Sharpe_Ratio"],
        xmin=tangency["ESG_Score"],
        xmax=recommended["ESG_Score"],
        colors=tangency_color,
        linestyles="--",
        linewidth=2.0,
        alpha=0.9,
    )
    ax.vlines(
        recommended["ESG_Score"],
        ymin=recommended["Sharpe_Ratio"],
        ymax=tangency["Sharpe_Ratio"],
        colors=guide_color,
        linestyles="-",
        linewidth=2.0,
        alpha=0.95,
    )

    add_end_label(ax, curve["ESG_Score"].iloc[-1], curve["Sharpe_Ratio"].iloc[-1], "Sharpe-ESG trade-off", frontier_color)
    add_end_label(
        ax,
        tangency["ESG_Score"],
        tangency["Sharpe_Ratio"],
        f"Tangency | SR {tangency['Sharpe_Ratio']:.3f}",
        tangency_color,
    )
    add_end_label(
        ax,
        recommended["ESG_Score"],
        recommended["Sharpe_Ratio"],
        f"Recommended | SR {recommended['Sharpe_Ratio']:.3f}",
        recommended_color,
    )

    sharpe_gap = tangency["Sharpe_Ratio"] - recommended["Sharpe_Ratio"]
    esg_gain = recommended["ESG_Score"] - tangency["ESG_Score"]
    ax.annotate(
        f"Green cost: {sharpe_gap:+.3f} SR\nESG gain: {esg_gain:+.2f}",
        (recommended["ESG_Score"], tangency["Sharpe_Ratio"]),
        textcoords="offset points",
        xytext=(14, -58),
        fontsize=10,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#f59e0b", alpha=0.98)
    )

    ax.set_xlabel("Portfolio ESG Score")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("ESG-Efficient Frontier (Sharpe-ESG Trade-off)")
    style_axis(ax)
    fig.tight_layout()
    return fig


def make_price_history_figure(prices, ticker1, ticker2):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    color1 = "#0071e3"
    color2 = "#5e5ce6"

    ax.plot(prices.index, prices[ticker1], linewidth=2.8, color=color1)
    ax.plot(prices.index, prices[ticker2], linewidth=2.8, color=color2)

    add_end_label(ax, prices.index[-1], prices[ticker1].iloc[-1], ticker1, color1)
    add_end_label(ax, prices.index[-1], prices[ticker2].iloc[-1], ticker2, color2)

    ax.set_title("Historical Price Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Closing Price")
    style_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Investor Survey")

q1 = st.sidebar.radio(
    "1. If your investment falls by 15% in one month, what would you most likely do?",
    [
        "Sell most of it immediately",
        "Sell part of it and wait",
        "Hold and wait for recovery",
        "Buy more at the lower price"
    ],
    index=2
)

q2 = st.sidebar.radio(
    "2. Which statement best reflects your investment approach?",
    [
        "I prefer stability even if returns are lower",
        "I want a balance between safety and growth",
        "I can tolerate volatility for better returns",
        "I actively seek higher returns despite high risk"
    ],
    index=1
)

q3 = st.sidebar.radio(
    "3. What level of annual loss would make you uncomfortable?",
    [
        "More than 5%",
        "More than 10%",
        "More than 20%",
        "More than 30%"
    ],
    index=1
)

q4 = st.sidebar.radio(
    "4. When two investments offer similar returns, how important is sustainability?",
    [
        "Not important",
        "Slightly important",
        "Moderately important",
        "Very important"
    ],
    index=2
)

q5 = st.sidebar.radio(
    "5. Would you accept a slightly lower return for stronger ESG characteristics?",
    [
        "No",
        "Only if the return difference is very small",
        "Yes, to some extent",
        "Yes, sustainability is a major priority"
    ],
    index=2
)

sin_stock_exclusions = st.sidebar.multiselect(
    "6. Exclude these sin industries",
    SIN_INDUSTRIES,
    default=[],
)

risk_scores = {
    "Sell most of it immediately": 10,
    "Sell part of it and wait": 7,
    "Hold and wait for recovery": 4,
    "Buy more at the lower price": 2,
    "I prefer stability even if returns are lower": 10,
    "I want a balance between safety and growth": 7,
    "I can tolerate volatility for better returns": 4,
    "I actively seek higher returns despite high risk": 2,
    "More than 5%": 10,
    "More than 10%": 7,
    "More than 20%": 4,
    "More than 30%": 2,
}

esg_scores = {
    "Not important": 1,
    "Slightly important": 3,
    "Moderately important": 6,
    "Very important": 9,
    "No": 1,
    "Only if the return difference is very small": 3,
    "Yes, to some extent": 6,
    "Yes, sustainability is a major priority": 9,
}

risk_total = risk_scores[q1] + risk_scores[q2] + risk_scores[q3]
esg_total = esg_scores[q4] + esg_scores[q5]

gamma = risk_total / 3
lambda_raw_avg = esg_total / 2
lambda_esg = lambda_raw_avg / 100


st.sidebar.header("Asset Inputs")
ticker1 = st.sidebar.text_input("Ticker for Asset 1", value="AAPL").strip().upper()
profile1 = fetch_ticker_profile(ticker1)
if ticker1:
    if profile1:
        st.sidebar.markdown(
            f"**{profile1['name']}**  \nSector: {profile1['sector']}  \nIndustry: {profile1['industry']}"
        )
        if is_sin_industry(profile1["industry"]):
            render_sin_warning()
    else:
        st.sidebar.markdown("Sector: Unavailable  \nIndustry: Unavailable")

ticker2 = st.sidebar.text_input("Ticker for Asset 2", value="MSFT").strip().upper()
profile2 = fetch_ticker_profile(ticker2)
if ticker2:
    if profile2:
        st.sidebar.markdown(
            f"**{profile2['name']}**  \nSector: {profile2['sector']}  \nIndustry: {profile2['industry']}"
        )
        if is_sin_industry(profile2["industry"]):
            render_sin_warning()
    else:
        st.sidebar.markdown("Sector: Unavailable  \nIndustry: Unavailable")

lookback_years = st.sidebar.slider(
    "Historical lookback period (years)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=1.0,
)
period = f"{int(lookback_years)}y"

esg1 = st.sidebar.slider(f"Manual ESG rating for {ticker1 or 'Asset 1'}", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
esg2 = st.sidebar.slider(f"Manual ESG rating for {ticker2 or 'Asset 2'}", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

investment_amount = st.sidebar.number_input("Total amount to invest (optional)", min_value=0.0, value=0.0, step=100.0)
risk_free_rate_pct = st.sidebar.slider("Risk-free rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_free_rate = risk_free_rate_pct / 100

run_button = st.sidebar.button("Run portfolio optimisation")


# =========================
# Main output
# =========================
if run_button:
    if not ticker1 or not ticker2:
        st.error("Please enter both tickers.")
    elif ticker1 == ticker2:
        st.error("Please choose two different tickers.")
    elif profile1 and profile1["industry"] in sin_stock_exclusions:
        st.error(f"{ticker1} belongs to an excluded sin industry: {profile1['industry']}.")
    elif profile2 and profile2["industry"] in sin_stock_exclusions:
        st.error(f"{ticker2} belongs to an excluded sin industry: {profile2['industry']}.")
    else:
        try:
            market_data = fetch_market_data(ticker1, ticker2, period)

            df = build_portfolio_table(
                r1=market_data["r1"],
                r2=market_data["r2"],
                sd1=market_data["sd1"],
                sd2=market_data["sd2"],
                corr=market_data["corr"],
                rf=risk_free_rate,
                esg1=esg1,
                esg2=esg2,
                gamma=gamma,
                lambda_esg=lambda_esg
            )

            recommended = select_recommended_portfolio(df)
            tangency = select_max_sharpe_portfolio(df)
            complete_portfolio = build_complete_portfolio(tangency, risk_free_rate, gamma)

            render_section_title("Investor Profile")
            c1, c2, c3, c4 = st.columns(4)
            render_metric_card(c1, "Risk attitude", classify_risk(gamma))
            render_metric_card(c2, "Sustainability profile", classify_esg(lambda_raw_avg))
            render_metric_card(c3, "Risk aversion score", f"{gamma:.2f}")
            render_metric_card(c4, "ESG preference score", f"{lambda_raw_avg:.2f} / 9")

            render_section_title("Market Data Summary")
            md = pd.DataFrame({
                "Metric": ["Expected annual return", "Annual volatility", "ESG score used"],
                ticker1: [market_data["r1"], market_data["sd1"], esg1],
                ticker2: [market_data["r2"], market_data["sd2"], esg2],
            })
            md[ticker1] = [f"{md.loc[0, ticker1]*100:.2f}%", f"{md.loc[1, ticker1]*100:.2f}%", f"{md.loc[2, ticker1]:.2f}"]
            md[ticker2] = [f"{md.loc[0, ticker2]*100:.2f}%", f"{md.loc[1, ticker2]*100:.2f}%", f"{md.loc[2, ticker2]:.2f}"]
            st.table(style_table(md))
            st.write(f"Correlation between {ticker1} and {ticker2}: **{market_data['corr']:.3f}**")
            st.write(f"Risk-free rate used: **{risk_free_rate * 100:.2f}%**")

            render_section_title("Tangency Portfolio")
            tangency_weights_df = pd.DataFrame({
                "Asset": [ticker1, ticker2],
                "Tangency Weight": [tangency["Weight_Asset1"], tangency["Weight_Asset2"]],
                "Amount if fully invested in tangency portfolio": [
                    investment_amount * tangency["Weight_Asset1"],
                    investment_amount * tangency["Weight_Asset2"]
                ]
            })
            tangency_weights_df["Tangency Weight"] = tangency_weights_df["Tangency Weight"].map(lambda x: f"{x*100:.2f}%")
            tangency_weights_df["Amount if fully invested in tangency portfolio"] = tangency_weights_df[
                "Amount if fully invested in tangency portfolio"
            ].map(lambda x: f"{x:,.2f}")
            st.table(style_table(tangency_weights_df))

            p1, p2, p3, p4 = st.columns(4)
            render_metric_card(p1, "Expected return", f"{tangency['Expected_Return']*100:.2f}%")
            render_metric_card(p2, "Volatility", f"{tangency['Risk_SD']*100:.2f}%")
            render_metric_card(p3, "Sharpe ratio", f"{tangency['Sharpe_Ratio']:.3f}")
            render_metric_card(p4, "ESG score", f"{tangency['ESG_Score']:.2f}")

            render_section_title("Recommended ESG Portfolio")
            recommended_weights_df = pd.DataFrame({
                "Asset": [ticker1, ticker2],
                "Recommended Weight": [recommended["Weight_Asset1"], recommended["Weight_Asset2"]],
                "Amount": [
                    investment_amount * recommended["Weight_Asset1"],
                    investment_amount * recommended["Weight_Asset2"]
                ]
            })
            recommended_weights_df["Recommended Weight"] = recommended_weights_df["Recommended Weight"].map(
                lambda x: f"{x*100:.2f}%"
            )
            recommended_weights_df["Amount"] = recommended_weights_df["Amount"].map(lambda x: f"{x:,.2f}")
            st.table(style_table(recommended_weights_df))

            p1, p2, p3, p4, p5 = st.columns(5)
            render_metric_card(p1, "Expected return", f"{recommended['Expected_Return']*100:.2f}%")
            render_metric_card(p2, "Volatility", f"{recommended['Risk_SD']*100:.2f}%")
            render_metric_card(p3, "ESG score", f"{recommended['ESG_Score']:.2f}")
            render_metric_card(p4, "Sharpe ratio", f"{recommended['Sharpe_Ratio']:.3f}")
            render_metric_card(p5, "ESG utility", f"{recommended['Utility']:.4f}")

            render_green_cost_card(
                sharpe_gap=tangency["Sharpe_Ratio"] - recommended["Sharpe_Ratio"],
                esg_gain=recommended["ESG_Score"] - tangency["ESG_Score"],
            )

            render_section_title("Complete Portfolio")
            complete_weights_df = pd.DataFrame({
                "Asset": [ticker1, ticker2, "Risk-free asset"],
                "Complete Portfolio Weight": [
                    complete_portfolio["Weight_Asset1"],
                    complete_portfolio["Weight_Asset2"],
                    complete_portfolio["weight_risk_free"]
                ],
                "Amount": [
                    investment_amount * complete_portfolio["Weight_Asset1"],
                    investment_amount * complete_portfolio["Weight_Asset2"],
                    investment_amount * complete_portfolio["weight_risk_free"]
                ]
            })
            complete_weights_df["Complete Portfolio Weight"] = complete_weights_df["Complete Portfolio Weight"].map(
                lambda x: f"{x*100:.2f}%"
            )
            complete_weights_df["Amount"] = complete_weights_df["Amount"].map(lambda x: f"{x:,.2f}")
            st.table(style_table(complete_weights_df))

            p1, p2, p3, p4, p5 = st.columns(5)
            render_metric_card(p1, "Tangency portfolio weight", f"{complete_portfolio['y']:.3f}")
            render_metric_card(p2, "Weight in risk-free asset", f"{complete_portfolio['weight_risk_free']:.3f}")
            render_metric_card(p3, "Expected Return", f"{complete_portfolio['Expected_Return']*100:.2f}%")
            render_metric_card(p4, "Volatility", f"{complete_portfolio['Risk_SD']*100:.2f}%")
            render_metric_card(p5, "Utility", f"{complete_portfolio['Utility']:.4f}")

            st.caption(
                "Complete portfolio formulas used: "
                "Expected Return = rf + y(E[Rt] - rf), Variance = y^2 Var(Rt), "
                "U = Expected Return - 0.5 * gamma * Variance."
            )

            render_section_title("Risky Portfolio Comparison")
            compare_df = pd.DataFrame({
                "Metric": [
                    "Weight in Asset 1",
                    "Weight in Asset 2",
                    "ESG score",
                    "ESG utility",
                    "Sharpe ratio",
                    "Expected return",
                    "Volatility",
                ],
                "Recommended ESG Portfolio": [
                    f"{recommended['Weight_Asset1']*100:.2f}%",
                    f"{recommended['Weight_Asset2']*100:.2f}%",
                    f"{recommended['ESG_Score']:.2f}",
                    f"{recommended['Utility']:.4f}",
                    f"{recommended['Sharpe_Ratio']:.3f}",
                    f"{recommended['Expected_Return']*100:.2f}%",
                    f"{recommended['Risk_SD']*100:.2f}%",
                ],
                "Tangency Portfolio": [
                    f"{tangency['Weight_Asset1']*100:.2f}%",
                    f"{tangency['Weight_Asset2']*100:.2f}%",
                    f"{tangency['ESG_Score']:.2f}",
                    "N/A",
                    f"{tangency['Sharpe_Ratio']:.3f}",
                    f"{tangency['Expected_Return']*100:.2f}%",
                    f"{tangency['Risk_SD']*100:.2f}%",
                ],
            })
            st.table(style_table(compare_df))

            render_section_title("Complete Portfolio Comparison")
            compare_df = pd.DataFrame({
                "Metric": [
                    "Weight in Asset 1",
                    "Weight in Asset 2",
                    "Weight in risk-free asset",
                    "Expected return",
                    "Volatility",
                    "Variance",
                    "Sharpe ratio",
                    "Utility",
                ],
                "Tangency Portfolio": [
                    f"{tangency['Weight_Asset1']*100:.2f}%",
                    f"{tangency['Weight_Asset2']*100:.2f}%",
                    "0.00%",
                    f"{tangency['Expected_Return']*100:.2f}%",
                    f"{tangency['Risk_SD']*100:.2f}%",
                    f"{tangency['Variance']:.4f}",
                    f"{tangency['Sharpe_Ratio']:.3f}",
                    "N/A",
                ],
                "Complete Portfolio": [
                    f"{complete_portfolio['Weight_Asset1']*100:.2f}%",
                    f"{complete_portfolio['Weight_Asset2']*100:.2f}%",
                    f"{complete_portfolio['weight_risk_free']*100:.2f}%",
                    f"{complete_portfolio['Expected_Return']*100:.2f}%",
                    f"{complete_portfolio['Risk_SD']*100:.2f}%",
                    f"{complete_portfolio['Variance']:.4f}",
                    f"{tangency['Sharpe_Ratio']:.3f}",
                    f"{complete_portfolio['Utility']:.4f}",
                ],
            })
            st.table(style_table(compare_df))

            render_section_title("Charts")
            fig0 = make_price_history_figure(market_data["prices"], ticker1, ticker2)
            st.pyplot(fig0)

            fig1 = make_frontier_figure(
                df, tangency, recommended, ticker1, ticker2,
                market_data["r1"], market_data["r2"], market_data["sd1"], market_data["sd2"]
            )
            st.pyplot(fig1)

            fig2 = make_cml_figure(
                df, tangency, complete_portfolio, risk_free_rate, ticker1, ticker2,
                market_data["r1"], market_data["r2"], market_data["sd1"], market_data["sd2"]
            )
            st.pyplot(fig2)

            fig3 = make_esg_tradeoff_figure(df, recommended)
            st.pyplot(fig3)

            fig4 = make_esg_efficient_frontier_figure(df, tangency, recommended)
            st.pyplot(fig4)

        except Exception as e:
            st.error(str(e))
else:
    st.info("Set your inputs in the sidebar, then click 'Run portfolio optimisation'.")
