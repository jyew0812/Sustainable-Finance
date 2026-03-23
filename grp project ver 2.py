import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import yfinance as yf
import streamlit as st


st.set_page_config(page_title="Tangency Portfolio Optimiser", layout="wide")


def inject_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {
                --bg: #f5f5f7;
                --card: rgba(255, 255, 255, 0.9);
                --border: rgba(15, 23, 42, 0.08);
                --text: #111827;
                --muted: #6b7280;
                --accent: #0071e3;
                --accent-2: #5e5ce6;
                --accent-3: #34c759;
                --shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(0, 113, 227, 0.08), transparent 28%),
                    radial-gradient(circle at top right, rgba(94, 92, 230, 0.08), transparent 24%),
                    linear-gradient(180deg, #fbfbfd 0%, #f2f4f7 100%);
                color: var(--text);
                font-family: 'Inter', sans-serif;
            }

            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0.72);
                backdrop-filter: blur(14px);
            }

            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.82);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] * {
                color: #111827 !important;
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2rem;
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.84));
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
                background: #e8f2ff;
                color: #0071e3;
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
                color: #111827;
            }

            .hero-subtitle {
                margin-top: 0.9rem;
                max-width: 760px;
                font-size: 1.02rem;
                line-height: 1.7;
                color: #6b7280;
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
                min-height: 160px;
            }

            [data-testid="stMetricLabel"] {
                color: #4b5563 !important;
                font-size: 0.98rem !important;
            }

            [data-testid="stMetricValue"] {
                color: #111827 !important;
                font-size: clamp(2rem, 3vw, 3.1rem) !important;
                line-height: 1.04 !important;
                white-space: normal !important;
                word-break: break-word !important;
                overflow-wrap: anywhere !important;
            }

            [data-testid="stTable"] {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            }

            .stAlert {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--border);
                color: #111827;
                border-radius: 20px;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            .stNumberInput > div > div,
            .stTextInput > div > div {
                border-radius: 14px !important;
                border-color: rgba(15, 23, 42, 0.08) !important;
                background: #ffffff !important;
                color: #111827 !important;
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
            .stNumberInput input {
                color: #111827 !important;
                -webkit-text-fill-color: #111827 !important;
                caret-color: #111827 !important;
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

            .section-title {
                font-size: 1.35rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                margin-top: 0.25rem;
                margin-bottom: 0.8rem;
                color: #111827;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Mean-Variance Portfolio Theory</div>
            <h1 class="hero-title">Tangency Portfolio Optimiser</h1>
            <p class="hero-subtitle">
                Compute the tangency portfolio from two risky assets, then combine it with the risk-free asset
                to build the complete portfolio that maximises investor utility under risk aversion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


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


def label_point(ax, x, y, text, color="#111827", dx=8, dy=8):
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.96),
    )


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
    annual_returns = returns.mean() * 252
    annual_cov = returns.cov() * 252

    return {
        "prices": prices,
        "returns": returns,
        "annual_returns": annual_returns,
        "annual_cov": annual_cov,
    }


def build_frontier(annual_returns, annual_cov, rf, ticker1, ticker2):
    weights = np.linspace(0, 1, 1001)
    rows = []

    for w1 in weights:
        weight_vector = np.array([w1, 1 - w1])
        expected_return = float(weight_vector @ annual_returns.values)
        variance = float(weight_vector @ annual_cov.values @ weight_vector)
        volatility = float(np.sqrt(variance))
        sharpe = (expected_return - rf) / volatility if volatility > 0 else np.nan

        rows.append(
            {
                ticker1: w1,
                ticker2: 1 - w1,
                "Expected Return": expected_return,
                "Variance": variance,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe,
            }
        )

    frontier = pd.DataFrame(rows).sort_values("Volatility").reset_index(drop=True)
    tangency = frontier.loc[frontier["Sharpe Ratio"].idxmax()]
    return frontier, tangency


def compute_complete_portfolio(tangency_row, rf, gamma):
    tangency_return = tangency_row["Expected Return"]
    tangency_variance = tangency_row["Variance"]
    tangency_volatility = tangency_row["Volatility"]
    tangency_sharpe = tangency_row["Sharpe Ratio"]

    y_star = (tangency_return - rf) / (gamma * tangency_variance)
    complete_return = rf + y_star * (tangency_return - rf)
    complete_variance = (y_star ** 2) * tangency_variance
    complete_volatility = np.sqrt(complete_variance)
    utility = complete_return - 0.5 * gamma * complete_variance

    return {
        "y_star": float(y_star),
        "risk_free_weight": float(1 - y_star),
        "expected_return": float(complete_return),
        "variance": float(complete_variance),
        "volatility": float(complete_volatility),
        "utility": float(utility),
        "tangency_return": float(tangency_return),
        "tangency_volatility": float(tangency_volatility),
        "tangency_sharpe": float(tangency_sharpe),
    }


def make_price_chart(prices, ticker1, ticker2):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    color1 = "#0071e3"
    color2 = "#5e5ce6"

    ax.plot(prices.index, prices[ticker1], linewidth=2.8, color=color1)
    ax.plot(prices.index, prices[ticker2], linewidth=2.8, color=color2)

    label_point(ax, prices.index[-1], prices[ticker1].iloc[-1], ticker1, color1, dx=8, dy=2)
    label_point(ax, prices.index[-1], prices[ticker2].iloc[-1], ticker2, color2, dx=8, dy=-18)

    ax.set_title("Historical Price Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Closing Price")
    style_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def make_frontier_chart(frontier, tangency_row, complete_portfolio, annual_returns, annual_cov, rf, ticker1, ticker2):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    frontier_x = frontier["Volatility"].values
    frontier_y = frontier["Expected Return"].values

    ax.plot(frontier_x, frontier_y, color="#0071e3", linewidth=3)
    ax.fill_between(frontier_x, frontier_y, color="#0071e3", alpha=0.08)

    tangency_x = tangency_row["Volatility"]
    tangency_y = tangency_row["Expected Return"]
    ax.scatter(tangency_x, tangency_y, s=180, marker="D", color="#34c759", edgecolors="white", linewidths=1.4)

    asset1_vol = np.sqrt(float(annual_cov.loc[ticker1, ticker1]))
    asset2_vol = np.sqrt(float(annual_cov.loc[ticker2, ticker2]))
    ax.scatter(
        [asset1_vol, asset2_vol],
        [annual_returns[ticker1], annual_returns[ticker2]],
        s=105,
        color="#9ca3af",
    )

    cal_y = np.linspace(0, max(1.6, complete_portfolio["y_star"] + 0.35), 200)
    cal_x = cal_y * tangency_x
    cal_returns = rf + cal_y * (tangency_y - rf)
    ax.plot(cal_x, cal_returns, color="#111827", linewidth=2.4, linestyle="--")

    complete_x = complete_portfolio["volatility"]
    complete_y = complete_portfolio["expected_return"]
    ax.scatter(0, rf, s=110, color="#f59e0b", edgecolors="white", linewidths=1.2)
    ax.scatter(complete_x, complete_y, s=170, color="#111827", edgecolors="white", linewidths=1.4)

    label_point(ax, asset1_vol, annual_returns[ticker1], ticker1, "#6b7280", dx=8, dy=10)
    label_point(ax, asset2_vol, annual_returns[ticker2], ticker2, "#6b7280", dx=8, dy=-18)
    label_point(ax, tangency_x, tangency_y, "Tangency", "#15803d", dx=10, dy=10)
    label_point(ax, complete_x, complete_y, "Complete", "#111827", dx=10, dy=10)
    label_point(ax, 0, rf, "Risk-free", "#b45309", dx=10, dy=10)

    ax.annotate(
        "CAL",
        (cal_x[-1], cal_returns[-1]),
        textcoords="offset points",
        xytext=(-28, -16),
        fontsize=10,
        color="#111827",
    )

    ax.set_title("Efficient Frontier and Capital Allocation Line")
    ax.set_xlabel("Portfolio Volatility")
    ax.set_ylabel("Expected Return")
    style_axis(ax, x_percent=True, y_percent=True)
    fig.tight_layout()
    return fig


inject_theme()
render_hero()

st.sidebar.header("Inputs")
ticker1 = st.sidebar.text_input("Ticker for Asset 1", value="AAPL").strip().upper()
ticker2 = st.sidebar.text_input("Ticker for Asset 2", value="MSFT").strip().upper()
gamma = st.sidebar.number_input("Gamma (risk aversion)", min_value=0.1, value=3.0, step=0.1)
risk_free_rate_pct = st.sidebar.slider("Risk-free rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

period_label = st.sidebar.selectbox(
    "Historical lookback period",
    ["6 months", "1 year", "3 years", "5 years"],
    index=1,
)
period_map = {
    "6 months": "6mo",
    "1 year": "1y",
    "3 years": "3y",
    "5 years": "5y",
}
period = period_map[period_label]
rf = risk_free_rate_pct / 100

run_button = st.sidebar.button("Run tangency portfolio analysis")


if run_button:
    if not ticker1 or not ticker2:
        st.error("Please enter both tickers.")
    elif ticker1 == ticker2:
        st.error("Please choose two different tickers.")
    else:
        try:
            market_data = fetch_market_data(ticker1, ticker2, period)
            annual_returns = market_data["annual_returns"]
            annual_cov = market_data["annual_cov"]

            frontier, tangency_row = build_frontier(annual_returns, annual_cov, rf, ticker1, ticker2)
            complete_portfolio = compute_complete_portfolio(tangency_row, rf, gamma)

            render_section_title("Annualised Market Inputs")
            summary_df = pd.DataFrame(
                {
                    "Metric": ["Expected return", "Volatility"],
                    ticker1: [
                        f"{annual_returns[ticker1] * 100:.2f}%",
                        f"{np.sqrt(annual_cov.loc[ticker1, ticker1]) * 100:.2f}%",
                    ],
                    ticker2: [
                        f"{annual_returns[ticker2] * 100:.2f}%",
                        f"{np.sqrt(annual_cov.loc[ticker2, ticker2]) * 100:.2f}%",
                    ],
                }
            )
            st.table(style_table(summary_df))

            cov_df = annual_cov.copy()
            cov_df.index.name = "Asset"
            cov_display = cov_df.reset_index()
            for col in [ticker1, ticker2]:
                cov_display[col] = cov_display[col].map(lambda x: f"{x:.6f}")
            st.table(style_table(cov_display))

            render_section_title("Tangency Portfolio")
            weight_col, metric_col = st.columns([1.05, 1.95])

            tangency_weights = pd.DataFrame(
                {
                    "Asset": [ticker1, ticker2],
                    "Tangency Weight": [
                        f"{tangency_row[ticker1] * 100:.2f}%",
                        f"{tangency_row[ticker2] * 100:.2f}%",
                    ],
                }
            )
            weight_col.table(style_table(tangency_weights))

            m1, m2, m3 = metric_col.columns(3)
            m1.metric("Tangency Return", f"{complete_portfolio['tangency_return'] * 100:.2f}%")
            m2.metric("Tangency Volatility", f"{complete_portfolio['tangency_volatility'] * 100:.2f}%")
            m3.metric("Tangency Sharpe Ratio", f"{complete_portfolio['tangency_sharpe']:.3f}")

            render_section_title("Complete Portfolio")
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal y in Tangency Portfolio", f"{complete_portfolio['y_star']:.3f}")
            c2.metric("Weight in Risk-Free Asset", f"{complete_portfolio['risk_free_weight']:.3f}")
            c3.metric("Utility", f"{complete_portfolio['utility']:.4f}")

            c4, c5 = st.columns(2)
            c4.metric("Complete Portfolio Return", f"{complete_portfolio['expected_return'] * 100:.2f}%")
            c5.metric("Complete Portfolio Volatility", f"{complete_portfolio['volatility'] * 100:.2f}%")

            if complete_portfolio["risk_free_weight"] < 0:
                st.info("A negative weight in the risk-free asset means the investor is borrowing at the risk-free rate to lever the tangency portfolio.")

            render_section_title("Charts")
            st.pyplot(make_price_chart(market_data["prices"], ticker1, ticker2))
            st.pyplot(make_frontier_chart(frontier, tangency_row, complete_portfolio, annual_returns, annual_cov, rf, ticker1, ticker2))

        except Exception as error:
            st.error(str(error))
else:
    st.info("Enter the tickers and portfolio inputs in the sidebar, then run the tangency portfolio analysis.")
