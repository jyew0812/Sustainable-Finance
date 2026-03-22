import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Sustainable Portfolio Recommender", layout="wide")
st.title("Sustainable Portfolio Recommender")
st.caption("Two-asset ESG portfolio optimiser using Yahoo Finance data and manual ESG ratings.")


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
    risk_free_rate = 0.02

    return {
        "prices": prices,
        "returns": returns,
        "r1": float(mean_returns[ticker1]),
        "r2": float(mean_returns[ticker2]),
        "sd1": float(volatilities[ticker1]),
        "sd2": float(volatilities[ticker2]),
        "corr": float(correlation),
        "rf": risk_free_rate
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


def utility_function(port_return_value, port_variance, port_esg_score, gamma, lambda_esg):
    return port_return_value - 0.5 * gamma * port_variance + lambda_esg * port_esg_score


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
        utility = utility_function(ret, var, esg_score, gamma, lambda_esg)

        rows.append({
            "Weight_Asset1": w1,
            "Weight_Asset2": w2,
            "Expected_Return": ret,
            "Variance": var,
            "Risk_SD": risk,
            "ESG_Score": esg_score,
            "Sharpe_Ratio": sharpe,
            "Utility": utility
        })

    return pd.DataFrame(rows)


def select_optimal_portfolio(df):
    return df.loc[df["Utility"].idxmax()]


def select_max_sharpe_portfolio(df):
    return df.loc[df["Sharpe_Ratio"].idxmax()]


def make_frontier_figure(df, optimal, max_sharpe, ticker1, ticker2, r1, r2, sd1, sd2):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Risk_SD"], df["Expected_Return"], linewidth=2, label="Feasible ESG Frontier")
    ax.scatter(optimal["Risk_SD"], optimal["Expected_Return"], s=120, marker="D", label="Recommended ESG Portfolio")
    ax.scatter(max_sharpe["Risk_SD"], max_sharpe["Expected_Return"], s=120, marker="X", label="Max-Sharpe Portfolio")
    ax.scatter([sd1, sd2], [r1, r2], s=100, label="Individual Assets")

    ax.annotate(ticker1, (sd1, r1), textcoords="offset points", xytext=(6, 6))
    ax.annotate(ticker2, (sd2, r2), textcoords="offset points", xytext=(6, 6))
    ax.annotate("Recommended", (optimal["Risk_SD"], optimal["Expected_Return"]), textcoords="offset points", xytext=(8, 8))
    ax.annotate("Max Sharpe", (max_sharpe["Risk_SD"], max_sharpe["Expected_Return"]), textcoords="offset points", xytext=(8, -14))

    ax.set_xlabel("Portfolio Risk (Standard Deviation)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("ESG Portfolio Frontier and Recommended Allocation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def make_esg_tradeoff_figure(df, optimal, max_sharpe):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["ESG_Score"], df["Expected_Return"], linewidth=2, label="Return-ESG Trade-off")
    ax.scatter(optimal["ESG_Score"], optimal["Expected_Return"], s=120, marker="D", label="Recommended ESG Portfolio")
    ax.scatter(max_sharpe["ESG_Score"], max_sharpe["Expected_Return"], s=120, marker="X", label="Max-Sharpe Portfolio")

    ax.annotate("Recommended", (optimal["ESG_Score"], optimal["Expected_Return"]), textcoords="offset points", xytext=(8, 8))
    ax.annotate("Max Sharpe", (max_sharpe["ESG_Score"], max_sharpe["Expected_Return"]), textcoords="offset points", xytext=(8, -14))

    ax.set_xlabel("Portfolio ESG Score")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("ESG and Return Trade-off")
    ax.grid(True)
    ax.legend()
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
ticker2 = st.sidebar.text_input("Ticker for Asset 2", value="MSFT").strip().upper()

period_label = st.sidebar.selectbox(
    "Historical lookback period",
    ["6 months", "1 year", "3 years", "5 years"],
    index=1
)

period_map = {
    "6 months": "6mo",
    "1 year": "1y",
    "3 years": "3y",
    "5 years": "5y"
}
period = period_map[period_label]

esg1 = st.sidebar.number_input(f"Manual ESG rating for {ticker1 or 'Asset 1'}", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
esg2 = st.sidebar.number_input(f"Manual ESG rating for {ticker2 or 'Asset 2'}", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

investment_amount = st.sidebar.number_input("Total amount to invest (optional)", min_value=0.0, value=10000.0, step=100.0)

run_button = st.sidebar.button("Run portfolio optimisation")


# =========================
# Main output
# =========================
if run_button:
    if not ticker1 or not ticker2:
        st.error("Please enter both tickers.")
    elif ticker1 == ticker2:
        st.error("Please choose two different tickers.")
    else:
        try:
            market_data = fetch_market_data(ticker1, ticker2, period)

            df = build_portfolio_table(
                r1=market_data["r1"],
                r2=market_data["r2"],
                sd1=market_data["sd1"],
                sd2=market_data["sd2"],
                corr=market_data["corr"],
                rf=market_data["rf"],
                esg1=esg1,
                esg2=esg2,
                gamma=gamma,
                lambda_esg=lambda_esg
            )

            optimal = select_optimal_portfolio(df)
            max_sharpe = select_max_sharpe_portfolio(df)

            st.subheader("Investor Profile")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk attitude", classify_risk(gamma))
            c2.metric("Sustainability profile", classify_esg(lambda_raw_avg))
            c3.metric("Risk aversion score", f"{gamma:.2f}")
            c4.metric("ESG preference score", f"{lambda_raw_avg:.2f} / 9")

            st.subheader("Market Data Summary")
            md = pd.DataFrame({
                "Metric": ["Expected annual return", "Annual volatility", "ESG score used"],
                ticker1: [market_data["r1"], market_data["sd1"], esg1],
                ticker2: [market_data["r2"], market_data["sd2"], esg2],
            })
            md[ticker1] = [f"{md.loc[0, ticker1]*100:.2f}%", f"{md.loc[1, ticker1]*100:.2f}%", f"{md.loc[2, ticker1]:.2f}"]
            md[ticker2] = [f"{md.loc[0, ticker2]*100:.2f}%", f"{md.loc[1, ticker2]*100:.2f}%", f"{md.loc[2, ticker2]:.2f}"]
            st.dataframe(md, use_container_width=True)
            st.write(f"Correlation between {ticker1} and {ticker2}: **{market_data['corr']:.3f}**")
            st.write(f"Risk-free rate used: **{market_data['rf']*100:.2f}%**")

            st.subheader("Recommended Portfolio")
            weights_df = pd.DataFrame({
                "Asset": [ticker1, ticker2],
                "Recommended Weight": [optimal["Weight_Asset1"], optimal["Weight_Asset2"]],
                "Amount": [
                    investment_amount * optimal["Weight_Asset1"],
                    investment_amount * optimal["Weight_Asset2"]
                ]
            })
            weights_df["Recommended Weight"] = weights_df["Recommended Weight"].map(lambda x: f"{x*100:.2f}%")
            weights_df["Amount"] = weights_df["Amount"].map(lambda x: f"{x:,.2f}")
            st.dataframe(weights_df, use_container_width=True)

            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Expected return", f"{optimal['Expected_Return']*100:.2f}%")
            p2.metric("Volatility", f"{optimal['Risk_SD']*100:.2f}%")
            p3.metric("ESG score", f"{optimal['ESG_Score']:.2f}")
            p4.metric("Sharpe ratio", f"{optimal['Sharpe_Ratio']:.3f}")
            p5.metric("Utility", f"{optimal['Utility']:.4f}")

            st.subheader("Comparison with Max-Sharpe Portfolio")
            compare_df = pd.DataFrame({
                "Metric": ["Weight in Asset 1", "Weight in Asset 2", "Expected return", "Volatility", "ESG score", "Sharpe ratio"],
                "Recommended ESG Portfolio": [
                    f"{optimal['Weight_Asset1']*100:.2f}%",
                    f"{optimal['Weight_Asset2']*100:.2f}%",
                    f"{optimal['Expected_Return']*100:.2f}%",
                    f"{optimal['Risk_SD']*100:.2f}%",
                    f"{optimal['ESG_Score']:.2f}",
                    f"{optimal['Sharpe_Ratio']:.3f}",
                ],
                "Max-Sharpe Portfolio": [
                    f"{max_sharpe['Weight_Asset1']*100:.2f}%",
                    f"{max_sharpe['Weight_Asset2']*100:.2f}%",
                    f"{max_sharpe['Expected_Return']*100:.2f}%",
                    f"{max_sharpe['Risk_SD']*100:.2f}%",
                    f"{max_sharpe['ESG_Score']:.2f}",
                    f"{max_sharpe['Sharpe_Ratio']:.3f}",
                ],
            })
            st.dataframe(compare_df, use_container_width=True)

            st.subheader("Charts")
            fig1 = make_frontier_figure(
                df, optimal, max_sharpe, ticker1, ticker2,
                market_data["r1"], market_data["r2"], market_data["sd1"], market_data["sd2"]
            )
            st.pyplot(fig1)

            fig2 = make_esg_tradeoff_figure(df, optimal, max_sharpe)
            st.pyplot(fig2)

        except Exception as e:
            st.error(str(e))
else:
    st.info("Set your inputs in the sidebar, then click 'Run portfolio optimisation'.")