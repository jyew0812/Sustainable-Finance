import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# Helper functions
# =========================
def ask_choice(question, options):
    print("\n" + question)
    for key, text in options.items():
        print(f"  {key}. {text}")

    while True:
        choice = input("Enter your choice: ").strip().upper()
        if choice in options:
            return choice
        print("Invalid choice. Please select one of the listed options.")


def ask_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def ask_ticker(prompt):
    while True:
        ticker = input(prompt).strip().upper()
        if ticker:
            return ticker
        print("Ticker cannot be empty.")


def ask_period():
    period_options = {
        "A": "6mo",
        "B": "1y",
        "C": "3y",
        "D": "5y"
    }

    labels = {
        "A": "6 months",
        "B": "1 year",
        "C": "3 years",
        "D": "5 years"
    }

    print("\nChoose historical lookback period:")
    for key, value in labels.items():
        print(f"  {key}. {value}")

    while True:
        choice = input("Enter your choice: ").strip()
        if choice in period_options:
            return period_options[choice], labels[choice]
        print("Invalid choice. Please select A, B, C, or D.")


# =========================
# Survey and scoring
# =========================
def run_investor_survey():
    print("=" * 60)
    print("SUSTAINABLE PORTFOLIO RECOMMENDER")
    print("=" * 60)
    print("Please complete this short survey.")
    print("Your answers will be used to estimate your risk attitude")
    print("and sustainability preference.\n")

    risk_questions = [
        {
            "question": "1. If your investment falls by 15% in a month, what would you most likely do?",
            "options": {
                "A": "Sell most of it immediately",
                "B": "Sell part of it and wait",
                "C": "Hold and wait for recovery",
                "D": "Buy more at the lower price"
            },
            "scores": {"A": 10, "B": 7, "C": 4, "D": 2}
        },
        {
            "question": "2. Which statement best reflects your investment approach?",
            "options": {
                "A": "I prefer stability even if returns are lower",
                "B": "I want a balance between safety and growth",
                "C": "I can tolerate volatility for better returns",
                "D": "I actively seek higher returns despite high risk"
            },
            "scores": {"A": 10, "B": 7, "C": 4, "D": 2}
        },
        {
            "question": "3. What level of annual loss would make you uncomfortable?",
            "options": {
                "A": "More than 5%",
                "B": "More than 10%",
                "C": "More than 20%",
                "D": "More than 30%"
            },
            "scores": {"A": 10, "B": 7, "C": 4, "D": 2}
        }
    ]

    esg_questions = [
        {
            "question": "4. When two investments offer similar returns, how important is sustainability?",
            "options": {
                "A": "Not important",
                "B": "Slightly important",
                "C": "Moderately important",
                "D": "Very important"
            },
            "scores": {"A": 1, "B": 3, "C": 6, "D": 9}
        },
        {
            "question": "5. Would you accept a slightly lower return for stronger ESG characteristics?",
            "options": {
                "A": "No",
                "B": "Only if the return difference is very small",
                "C": "Yes, to some extent",
                "D": "Yes, sustainability is a major priority"
            },
            "scores": {"A": 1, "B": 3, "C": 6, "D": 9}
        }
    ]

    risk_total = 0
    esg_total = 0

    print("SECTION 1: RISK ATTITUDE")
    for q in risk_questions:
        ans = ask_choice(q["question"], q["options"])
        risk_total += q["scores"][ans]

    print("\nSECTION 2: SUSTAINABILITY PREFERENCES")
    for q in esg_questions:
        ans = ask_choice(q["question"], q["options"])
        esg_total += q["scores"][ans]

    gamma = risk_total / len(risk_questions)          # risk aversion parameter
    lambda_raw = esg_total / len(esg_questions)       # 1 to 9 scale
    lambda_esg = lambda_raw / 100                     # scaled to avoid dominating utility

    return gamma, lambda_esg, risk_total, esg_total


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


# =========================
# Data fetching
# =========================
def fetch_market_data(ticker1, ticker2, period):
    print("\nDownloading market data from Yahoo Finance...")

    data = yf.download(
        [ticker1, ticker2],
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No data was downloaded. Check the tickers and try again.")

    if "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        raise ValueError("Could not retrieve both tickers properly. Please try different tickers.")

    prices = prices[[ticker1, ticker2]].dropna()

    if prices.empty or len(prices) < 30:
        raise ValueError("Insufficient price history after cleaning. Please choose different tickers or period.")

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


# =========================
# Portfolio calculations
# =========================
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


# =========================
# Output functions
# =========================
def print_market_summary(ticker1, ticker2, market_data):
    print("\n" + "=" * 60)
    print("MARKET DATA SUMMARY")
    print("=" * 60)
    print(f"{ticker1}:")
    print(f"  Expected annual return: {market_data['r1'] * 100:.2f}%")
    print(f"  Annual volatility:      {market_data['sd1'] * 100:.2f}%")

    print(f"\n{ticker2}:")
    print(f"  Expected annual return: {market_data['r2'] * 100:.2f}%")
    print(f"  Annual volatility:      {market_data['sd2'] * 100:.2f}%")

    print(f"\nCorrelation between {ticker1} and {ticker2}: {market_data['corr']:.3f}")
    print(f"Risk-free rate used: {market_data['rf'] * 100:.2f}%")


def print_recommendation(
    ticker1,
    ticker2,
    optimal,
    max_sharpe,
    gamma,
    lambda_raw_avg,
    investment_amount=None
):
    risk_profile = classify_risk(gamma)
    esg_profile = classify_esg(lambda_raw_avg)

    print("\n" + "=" * 60)
    print("INVESTOR PROFILE")
    print("=" * 60)
    print(f"Risk attitude:           {risk_profile}")
    print(f"Sustainability profile:  {esg_profile}")
    print(f"Risk aversion score:     {gamma:.2f}")
    print(f"ESG preference score:    {lambda_raw_avg:.2f} / 9")

    print("\n" + "=" * 60)
    print("RECOMMENDED ESG PORTFOLIO")
    print("=" * 60)
    print(f"Weight in {ticker1}: {optimal['Weight_Asset1'] * 100:.2f}%")
    print(f"Weight in {ticker2}: {optimal['Weight_Asset2'] * 100:.2f}%")

    print("\nPortfolio characteristics:")
    print(f"Expected annual return:  {optimal['Expected_Return'] * 100:.2f}%")
    print(f"Annual volatility:       {optimal['Risk_SD'] * 100:.2f}%")
    print(f"Portfolio ESG score:     {optimal['ESG_Score']:.2f}")
    print(f"Sharpe ratio:            {optimal['Sharpe_Ratio']:.3f}")
    print(f"Utility score:           {optimal['Utility']:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH MAX-SHARPE PORTFOLIO")
    print("=" * 60)
    print(f"Max-Sharpe weight in {ticker1}: {max_sharpe['Weight_Asset1'] * 100:.2f}%")
    print(f"Max-Sharpe weight in {ticker2}: {max_sharpe['Weight_Asset2'] * 100:.2f}%")
    print(f"Max-Sharpe expected return:     {max_sharpe['Expected_Return'] * 100:.2f}%")
    print(f"Max-Sharpe volatility:          {max_sharpe['Risk_SD'] * 100:.2f}%")
    print(f"Max-Sharpe ESG score:           {max_sharpe['ESG_Score']:.2f}")
    print(f"Max-Sharpe ratio:               {max_sharpe['Sharpe_Ratio']:.3f}")

    if investment_amount is not None:
        alloc1 = investment_amount * optimal["Weight_Asset1"]
        alloc2 = investment_amount * optimal["Weight_Asset2"]

        print("\n" + "=" * 60)
        print("CAPITAL ALLOCATION")
        print("=" * 60)
        print(f"Total investment amount: {investment_amount:,.2f}")
        print(f"Allocate to {ticker1}:    {alloc1:,.2f}")
        print(f"Allocate to {ticker2}:    {alloc2:,.2f}")

    print("\nInterpretation:")
    print(
        "The recommended portfolio is the allocation that maximises ESG-adjusted utility, "
        "taking into account both financial performance and your sustainability preference."
    )


def print_weights_table(ticker1, ticker2, optimal):
    weights_table = pd.DataFrame({
        "Asset": [ticker1, ticker2],
        "Recommended Weight": [optimal["Weight_Asset1"], optimal["Weight_Asset2"]]
    })

    print("\n" + "=" * 60)
    print("TABLE OF RECOMMENDED STOCK WEIGHTS")
    print("=" * 60)
    print(weights_table.to_string(index=False, formatters={
        "Recommended Weight": lambda x: f"{x * 100:.2f}%"
    }))


def plot_frontier(df, optimal, max_sharpe, ticker1, ticker2, r1, r2, sd1, sd2):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Risk_SD"], df["Expected_Return"], linewidth=2, label="Feasible ESG Frontier")
    plt.scatter(optimal["Risk_SD"], optimal["Expected_Return"], s=120, marker="D", label="Recommended ESG Portfolio")
    plt.scatter(max_sharpe["Risk_SD"], max_sharpe["Expected_Return"], s=120, marker="X", label="Max-Sharpe Portfolio")
    plt.scatter([sd1, sd2], [r1, r2], s=100, label="Individual Assets")

    plt.annotate(ticker1, (sd1, r1), textcoords="offset points", xytext=(6, 6))
    plt.annotate(ticker2, (sd2, r2), textcoords="offset points", xytext=(6, 6))
    plt.annotate("Recommended", (optimal["Risk_SD"], optimal["Expected_Return"]), textcoords="offset points", xytext=(8, 8))
    plt.annotate("Max Sharpe", (max_sharpe["Risk_SD"], max_sharpe["Expected_Return"]), textcoords="offset points", xytext=(8, -14))

    plt.xlabel("Portfolio Risk (Standard Deviation)")
    plt.ylabel("Expected Annual Return")
    plt.title("ESG Portfolio Frontier and Recommended Allocation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_esg_tradeoff(df, optimal, max_sharpe):
    plt.figure(figsize=(10, 6))
    plt.plot(df["ESG_Score"], df["Expected_Return"], linewidth=2, label="Return-ESG Trade-off")
    plt.scatter(optimal["ESG_Score"], optimal["Expected_Return"], s=120, marker="D", label="Recommended ESG Portfolio")
    plt.scatter(max_sharpe["ESG_Score"], max_sharpe["Expected_Return"], s=120, marker="X", label="Max-Sharpe Portfolio")

    plt.annotate("Recommended", (optimal["ESG_Score"], optimal["Expected_Return"]), textcoords="offset points", xytext=(8, 8))
    plt.annotate("Max Sharpe", (max_sharpe["ESG_Score"], max_sharpe["Expected_Return"]), textcoords="offset points", xytext=(8, -14))

    plt.xlabel("Portfolio ESG Score")
    plt.ylabel("Expected Annual Return")
    plt.title("ESG and Return Trade-off")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# Main program
# =========================
def main():
    try:
        gamma, lambda_esg, risk_total, esg_total = run_investor_survey()
        lambda_raw_avg = esg_total / 2

        print("\n" + "=" * 60)
        print("ASSET INPUT")
        print("=" * 60)

        ticker1 = ask_ticker("Enter ticker for Asset 1: ")
        ticker2 = ask_ticker("Enter ticker for Asset 2: ")

        while ticker1 == ticker2:
            print("Please choose two different tickers.")
            ticker2 = ask_ticker("Enter ticker for Asset 2: ")

        esg1 = ask_float(f"Enter ESG rating for {ticker1} (0 to 100): ", 0, 100)
        esg2 = ask_float(f"Enter ESG rating for {ticker2} (0 to 100): ", 0, 100)

        period, period_label = ask_period()
        print(f"\nSelected lookback period: {period_label}")

        investment_amount = input("\nEnter total amount to invest (or press Enter to skip): ").strip()
        if investment_amount == "":
            investment_amount = None
        else:
            investment_amount = float(investment_amount)
            if investment_amount < 0:
                raise ValueError("Investment amount cannot be negative.")

        market_data = fetch_market_data(ticker1, ticker2, period)

        print_market_summary(ticker1, ticker2, market_data)

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

        print_weights_table(ticker1, ticker2, optimal)
        print_recommendation(
            ticker1=ticker1,
            ticker2=ticker2,
            optimal=optimal,
            max_sharpe=max_sharpe,
            gamma=gamma,
            lambda_raw_avg=lambda_raw_avg,
            investment_amount=investment_amount
        )

        plot_frontier(
            df=df,
            optimal=optimal,
            max_sharpe=max_sharpe,
            ticker1=ticker1,
            ticker2=ticker2,
            r1=market_data["r1"],
            r2=market_data["r2"],
            sd1=market_data["sd1"],
            sd2=market_data["sd2"]
        )

        plot_esg_tradeoff(df, optimal, max_sharpe)

    except Exception as e:
        print("\nAn error occurred:")
        print(e)
        print("Please review your inputs and try again.")


if __name__ == "__main__":
    main()