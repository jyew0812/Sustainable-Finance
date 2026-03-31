import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
import yfinance as yf

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


st.set_page_config(page_title="Multiple Asset ESG Optimiser", layout="wide")
st.title("Multiple Asset ESG Optimiser (10 Risky Assets)")
st.write("Matrix-based portfolio optimisation with no short-selling and a complete portfolio.")


def classify_risk(gamma):
    if gamma >= 8:
        return "Defensive"
    if gamma >= 5:
        return "Balanced"
    return "Growth-Oriented"


def classify_esg(lambda_raw_avg):
    if lambda_raw_avg >= 7:
        return "Sustainability-Led"
    if lambda_raw_avg >= 4:
        return "ESG-Aware"
    return "Low ESG Priority"


@st.cache_data(show_spinner=False)
def fetch_market_data(tickers, period):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("No data was downloaded. Check the tickers and try again.")

    prices = data["Close"].copy() if "Close" in data.columns else data.copy()
    if isinstance(prices, pd.Series):
        raise ValueError("Could not retrieve all tickers properly.")

    prices = prices[tickers].dropna()
    if prices.empty or len(prices) < 60:
        raise ValueError("Insufficient price history. Choose a longer period or different tickers.")

    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252

    return {
        "prices": prices,
        "returns": returns,
        "mu": mu,
        "sigma": sigma,
        "tickers": tickers,
    }


def portfolio_metrics(weights, mu, sigma):
    expected_return = float(weights @ mu)
    variance = float(weights @ sigma @ weights)
    risk = float(np.sqrt(max(variance, 0.0)))
    return expected_return, variance, risk


def sharpe_ratio(weights, mu, sigma, rf):
    expected_return, _, risk = portfolio_metrics(weights, mu, sigma)
    if risk <= 0:
        return np.nan
    return (expected_return - rf) / risk


def esg_score(weights, esg_vector):
    return float(weights @ esg_vector)


def utility_risky(weights, mu, sigma, gamma, lambda_esg, esg_vector):
    expected_return, variance, _ = portfolio_metrics(weights, mu, sigma)
    return expected_return - 0.5 * gamma * variance + lambda_esg * esg_score(weights, esg_vector)


def _random_feasible_weights(n_assets, n_samples=30000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_assets), size=n_samples)


def optimise_tangency(mu, sigma, rf):
    n = len(mu)

    if SCIPY_AVAILABLE:
        w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            sr = sharpe_ratio(w, mu, sigma, rf)
            return 1e6 if np.isnan(sr) else -sr

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success:
            w = np.clip(result.x, 0, 1)
            w = w / w.sum()
            return w

    weights = _random_feasible_weights(n)
    sharpe_values = np.array([sharpe_ratio(w, mu, sigma, rf) for w in weights])
    idx = int(np.nanargmax(sharpe_values))
    return weights[idx]


def optimise_recommended_esg(mu, sigma, gamma, lambda_esg, esg_vector):
    n = len(mu)

    if SCIPY_AVAILABLE:
        w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            return -utility_risky(w, mu, sigma, gamma, lambda_esg, esg_vector)

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success:
            w = np.clip(result.x, 0, 1)
            w = w / w.sum()
            return w

    weights = _random_feasible_weights(n)
    util_values = np.array([utility_risky(w, mu, sigma, gamma, lambda_esg, esg_vector) for w in weights])
    idx = int(np.argmax(util_values))
    return weights[idx]


def build_complete_portfolio(tangency_weights, mu, sigma, rf, gamma):
    ret_t, var_t, risk_t = portfolio_metrics(tangency_weights, mu, sigma)
    if var_t <= 0:
        raise ValueError("Tangency variance is non-positive. Cannot build complete portfolio.")

    y = (ret_t - rf) / (gamma * var_t)
    expected_complete = rf + y * (ret_t - rf)
    variance_complete = (y ** 2) * var_t
    risk_complete = np.sqrt(max(variance_complete, 0.0))
    utility_complete = expected_complete - 0.5 * gamma * variance_complete

    return {
        "y": float(y),
        "weight_risk_free": float(1 - y),
        "Expected_Return": float(expected_complete),
        "Variance": float(variance_complete),
        "Risk_SD": float(risk_complete),
        "Utility": float(utility_complete),
        "Tangency_Return": float(ret_t),
        "Tangency_Variance": float(var_t),
        "Tangency_Risk": float(risk_t),
    }


def style_axis(ax, x_percent=False, y_percent=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#374151", labelsize=10)
    ax.grid(True, color="#e5e7eb", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.title.set_color("#111827")
    ax.xaxis.label.set_color("#374151")
    ax.yaxis.label.set_color("#374151")
    if x_percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    if y_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))


def make_risk_return_figure(mu, sigma, rf, tangency_w, recommended_w):
    n_assets = len(mu)
    cloud = _random_feasible_weights(n_assets, n_samples=7000, seed=7)
    risks = np.array([portfolio_metrics(w, mu, sigma)[2] for w in cloud])
    returns = np.array([portfolio_metrics(w, mu, sigma)[0] for w in cloud])

    tangency_ret, _, tangency_risk = portfolio_metrics(tangency_w, mu, sigma)
    rec_ret, _, rec_risk = portfolio_metrics(recommended_w, mu, sigma)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.scatter(risks, returns, s=8, alpha=0.18, color="#94a3b8")
    ax.scatter([tangency_risk], [tangency_ret], s=160, marker="X", color="#111827")
    ax.scatter([rec_risk], [rec_ret], s=120, marker="D", color="#34c759", edgecolors="white", linewidths=1.2)

    ax.annotate("Tangency", (tangency_risk, tangency_ret), textcoords="offset points", xytext=(8, -14))
    ax.annotate("Recommended", (rec_risk, rec_ret), textcoords="offset points", xytext=(8, 8))

    ax.set_title("Risk-Return Feasible Set (10 Assets)")
    ax.set_xlabel("Portfolio Risk (Standard Deviation)")
    ax.set_ylabel("Expected Annual Return")
    style_axis(ax, x_percent=True, y_percent=True)
    return fig


def make_cml_figure(mu, sigma, rf, tangency_w, complete):
    n_assets = len(mu)
    cloud = _random_feasible_weights(n_assets, n_samples=6000, seed=9)
    risks = np.array([portfolio_metrics(w, mu, sigma)[2] for w in cloud])
    returns = np.array([portfolio_metrics(w, mu, sigma)[0] for w in cloud])

    tangency_ret = complete["Tangency_Return"]
    tangency_risk = complete["Tangency_Risk"]
    sharpe_t = (tangency_ret - rf) / tangency_risk if tangency_risk > 0 else 0.0

    x_max = max(np.max(risks), complete["Risk_SD"], tangency_risk) * 1.15
    cml_x = np.linspace(0, x_max, 300)
    cml_y = rf + sharpe_t * cml_x

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.scatter(risks, returns, s=8, alpha=0.15, color="#cbd5e1")
    ax.plot(cml_x, cml_y, linewidth=2.6, color="#34c759", linestyle="--")
    ax.scatter([0], [rf], s=90, color="#5e5ce6")
    ax.scatter([tangency_risk], [tangency_ret], s=155, marker="X", color="#111827")
    ax.scatter([complete["Risk_SD"]], [complete["Expected_Return"]], s=120, marker="D", color="#ff9f0a", edgecolors="white")

    ax.annotate("Risk-free", (0, rf), textcoords="offset points", xytext=(8, 8))
    ax.annotate("Tangency", (tangency_risk, tangency_ret), textcoords="offset points", xytext=(8, -14))
    ax.annotate("Complete", (complete["Risk_SD"], complete["Expected_Return"]), textcoords="offset points", xytext=(8, 8))

    ax.set_title("Capital Market Line and Complete Portfolio")
    ax.set_xlabel("Portfolio Risk (Standard Deviation)")
    ax.set_ylabel("Expected Annual Return")
    style_axis(ax, x_percent=True, y_percent=True)
    return fig


def make_esg_sharpe_figure(mu, sigma, rf, esg_vector, tangency_w, recommended_w):
    n_assets = len(mu)
    cloud = _random_feasible_weights(n_assets, n_samples=7000, seed=21)
    esg_vals = np.array([esg_score(w, esg_vector) for w in cloud])
    sharpe_vals = np.array([sharpe_ratio(w, mu, sigma, rf) for w in cloud])

    tangency_esg = esg_score(tangency_w, esg_vector)
    tangency_sharpe = sharpe_ratio(tangency_w, mu, sigma, rf)
    rec_esg = esg_score(recommended_w, esg_vector)
    rec_sharpe = sharpe_ratio(recommended_w, mu, sigma, rf)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.scatter(esg_vals, sharpe_vals, s=8, alpha=0.18, color="#94a3b8")
    ax.scatter([tangency_esg], [tangency_sharpe], s=190, marker="*", color="#ff6b6b", edgecolors="white", linewidths=0.7)
    ax.scatter([rec_esg], [rec_sharpe], s=130, marker="D", color="#34c759", edgecolors="white", linewidths=1.0)

    ax.hlines(tangency_sharpe, xmin=min(tangency_esg, rec_esg), xmax=max(tangency_esg, rec_esg), colors="#ff6b6b", linestyles="--", linewidth=2)
    ax.vlines(rec_esg, ymin=min(tangency_sharpe, rec_sharpe), ymax=max(tangency_sharpe, rec_sharpe), colors="#f59e0b", linewidth=2)

    sharpe_gap = tangency_sharpe - rec_sharpe
    esg_gain = rec_esg - tangency_esg
    ax.annotate(
        f"Green cost: {sharpe_gap:+.3f} SR\nESG gain: {esg_gain:+.2f}",
        (rec_esg, tangency_sharpe),
        textcoords="offset points",
        xytext=(12, -52),
        fontsize=10,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#f59e0b", alpha=0.98),
    )

    ax.annotate("Tangency", (tangency_esg, tangency_sharpe), textcoords="offset points", xytext=(8, 8))
    ax.annotate("Recommended", (rec_esg, rec_sharpe), textcoords="offset points", xytext=(8, -16))

    ax.set_title("ESG-Efficient Frontier (Sharpe-ESG Trade-off)")
    ax.set_xlabel("Portfolio ESG Score")
    ax.set_ylabel("Sharpe Ratio")
    style_axis(ax)
    return fig


st.sidebar.header("Investor Survey")
q1 = st.sidebar.radio(
    "1. If your investment falls by 15% in one month, what would you most likely do?",
    [
        "Sell most of it immediately",
        "Sell part of it and wait",
        "Hold and wait for recovery",
        "Buy more at the lower price",
    ],
    index=2,
)
q2 = st.sidebar.radio(
    "2. Which statement best reflects your investment approach?",
    [
        "I prefer stability even if returns are lower",
        "I want a balance between safety and growth",
        "I can tolerate volatility for better returns",
        "I actively seek higher returns despite high risk",
    ],
    index=1,
)
q3 = st.sidebar.radio(
    "3. What level of annual loss would make you uncomfortable?",
    ["More than 5%", "More than 10%", "More than 20%", "More than 30%"],
    index=1,
)
q4 = st.sidebar.radio(
    "4. When two investments offer similar returns, how important is sustainability?",
    ["Not important", "Slightly important", "Moderately important", "Very important"],
    index=2,
)
q5 = st.sidebar.radio(
    "5. Would you accept a slightly lower return for stronger ESG characteristics?",
    ["No", "Only if the return difference is very small", "Yes, to some extent", "Yes, sustainability is a major priority"],
    index=2,
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

gamma = (risk_scores[q1] + risk_scores[q2] + risk_scores[q3]) / 3
lambda_raw_avg = (esg_scores[q4] + esg_scores[q5]) / 2
lambda_esg = lambda_raw_avg / 100.0


st.sidebar.header("Asset Inputs (10 Risky Assets)")
default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "PG"]
ticker_values = []
for i in range(10):
    ticker_values.append(st.sidebar.text_input(f"Ticker {i + 1}", value=default_tickers[i]).strip().upper())

lookback_years = st.sidebar.slider("Historical lookback period (years)", min_value=1, max_value=10, value=3, step=1)
risk_free_rate_pct = st.sidebar.slider("Risk-free rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
investment_amount = st.sidebar.number_input("Total amount to invest", min_value=0.0, value=0.0, step=100.0)

st.sidebar.markdown("### ESG Scores (0-100)")
esg_scores_inputs = []
for i, tk in enumerate(ticker_values):
    label = tk if tk else f"Ticker {i + 1}"
    esg_scores_inputs.append(st.sidebar.slider(f"ESG for {label}", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key=f"esg_{i}"))

run_button = st.sidebar.button("Run portfolio optimisation")


if run_button:
    if any(not tk for tk in ticker_values):
        st.error("Please fill all 10 ticker fields.")
    elif len(set(ticker_values)) != 10:
        st.error("Please use 10 unique tickers.")
    else:
        try:
            period = f"{lookback_years}y"
            rf = risk_free_rate_pct / 100.0
            market_data = fetch_market_data(ticker_values, period)
            mu = market_data["mu"]
            sigma = market_data["sigma"]
            esg_vector = np.array(esg_scores_inputs)

            tangency_w = optimise_tangency(mu, sigma, rf)
            recommended_w = optimise_recommended_esg(mu, sigma, gamma, lambda_esg, esg_vector)
            complete = build_complete_portfolio(tangency_w, mu, sigma, rf, gamma)

            tangency_ret, tangency_var, tangency_risk = portfolio_metrics(tangency_w, mu, sigma)
            recommended_ret, recommended_var, recommended_risk = portfolio_metrics(recommended_w, mu, sigma)
            tangency_sr = sharpe_ratio(tangency_w, mu, sigma, rf)
            recommended_sr = sharpe_ratio(recommended_w, mu, sigma, rf)
            tangency_esg = esg_score(tangency_w, esg_vector)
            recommended_esg = esg_score(recommended_w, esg_vector)

            st.subheader("Investor Profile")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk attitude", classify_risk(gamma))
            c2.metric("Sustainability profile", classify_esg(lambda_raw_avg))
            c3.metric("Risk aversion score (gamma)", f"{gamma:.2f}")
            c4.metric("ESG preference score", f"{lambda_raw_avg:.2f} / 9")

            st.subheader("Tangency Portfolio (10 Risky Assets)")
            tangency_df = pd.DataFrame({
                "Ticker": ticker_values,
                "Weight": tangency_w,
                "Amount": tangency_w * investment_amount,
            })
            st.dataframe(tangency_df.style.format({"Weight": "{:.2%}", "Amount": "{:,.2f}"}), use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected return", f"{tangency_ret*100:.2f}%")
            c2.metric("Volatility", f"{tangency_risk*100:.2f}%")
            c3.metric("Sharpe ratio", f"{tangency_sr:.3f}")
            c4.metric("ESG score", f"{tangency_esg:.2f}")

            st.subheader("Recommended ESG Portfolio (10 Risky Assets)")
            rec_df = pd.DataFrame({
                "Ticker": ticker_values,
                "Weight": recommended_w,
                "Amount": recommended_w * investment_amount,
            })
            st.dataframe(rec_df.style.format({"Weight": "{:.2%}", "Amount": "{:,.2f}"}), use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Expected return", f"{recommended_ret*100:.2f}%")
            c2.metric("Volatility", f"{recommended_risk*100:.2f}%")
            c3.metric("Sharpe ratio", f"{recommended_sr:.3f}")
            c4.metric("ESG score", f"{recommended_esg:.2f}")
            c5.metric("Green cost (SR diff)", f"{(tangency_sr - recommended_sr):+.3f}")

            st.subheader("Complete Portfolio")
            complete_weights_risky = complete["y"] * tangency_w
            complete_df = pd.DataFrame({
                "Ticker": ticker_values + ["Risk-free asset"],
                "Weight": np.append(complete_weights_risky, complete["weight_risk_free"]),
                "Amount": np.append(complete_weights_risky, complete["weight_risk_free"]) * investment_amount,
            })
            st.dataframe(complete_df.style.format({"Weight": "{:.2%}", "Amount": "{:,.2f}"}), use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("y in tangency portfolio", f"{complete['y']:.3f}")
            c2.metric("Weight in risk-free", f"{complete['weight_risk_free']:.3f}")
            c3.metric("Expected Return", f"{complete['Expected_Return']*100:.2f}%")
            c4.metric("Volatility", f"{complete['Risk_SD']*100:.2f}%")
            c5.metric("Utility", f"{complete['Utility']:.4f}")

            st.caption(
                "Matrix formulas used: E[Rp] = w^T mu, Var(Rp) = w^T Sigma w, "
                "Sharpe = (E[Rp]-rf)/sqrt(Var). "
                "Complete portfolio: y = (E[Rt]-rf)/(gamma*Var(Rt)), "
                "E[Rc]=rf+y(E[Rt]-rf), Var(Rc)=y^2 Var(Rt)."
            )

            st.subheader("Charts")
            st.pyplot(make_risk_return_figure(mu, sigma, rf, tangency_w, recommended_w))
            st.pyplot(make_cml_figure(mu, sigma, rf, tangency_w, complete))
            st.pyplot(make_esg_sharpe_figure(mu, sigma, rf, esg_vector, tangency_w, recommended_w))

            if not SCIPY_AVAILABLE:
                st.warning("SciPy was not available. Optimisation used a random-feasible-search fallback.")

        except Exception as e:
            st.error(str(e))
else:
    st.info("Set inputs in the sidebar and click 'Run portfolio optimisation'.")
