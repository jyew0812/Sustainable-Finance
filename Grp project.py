import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ask_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Please enter a value no lower than {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Please enter a value no higher than {max_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def classify_risk(gamma):
    if gamma <= 3:
        return "Aggressive"
    elif gamma <= 7:
        return "Balanced"
    else:
        return "Conservative"

def classify_esg(esg_pref):
    if esg_pref <= 3:
        return "Low ESG focus"
    elif esg_pref <= 7:
        return "Moderate ESG focus"
    else:
        return "Strong ESG focus"

def portfolio_return(w1, r1, r2):
    return w1 * r1 + (1 - w1) * r2

def portfolio_variance(w1, sd1, sd2, corr):
    w2 = 1 - w1
    return (w1**2) * (sd1**2) + (w2**2) * (sd2**2) + 2 * w1 * w2 * corr * sd1 * sd2

def portfolio_esg(w1, esg1, esg2):
    return w1 * esg1 + (1 - w1) * esg2

def utility_function(port_ret, port_var, port_esg, gamma, lam):
    return port_ret - 0.5 * gamma * port_var + lam * port_esg

print("SUSTAINABLE PORTFOLIO RECOMMENDER")
print("-" * 40)

gamma = ask_float("How sensitive are you to investment risk? Enter 1 to 10: ", 1, 10)
esg_pref_input = ask_float("How important is ESG to you? Enter 0 to 10: ", 0, 10)

# Scale ESG preference so it does not dominate utility numerically
lam = esg_pref_input / 100

print("\nEnter details for Asset A")
r1 = ask_float("Expected annual return (%) : ") / 100
sd1 = ask_float("Annual volatility / standard deviation (%) : ", 0) / 100
esg1 = ask_float("ESG rating (1 to 6) : ", 1, 6)

print("\nEnter details for Asset B")
r2 = ask_float("Expected annual return (%) : ") / 100
sd2 = ask_float("Annual volatility / standard deviation (%) : ", 0) / 100
esg2 = ask_float("ESG rating (1 to 6) : ", 1, 6)

corr = ask_float("\nCorrelation between Asset A and Asset B (-1 to 1) : ", -1, 1)
rf = ask_float("Risk-free rate (%) : ", 0) / 100

risk_profile = classify_risk(gamma)
esg_profile = classify_esg(esg_pref_input)

weights = np.linspace(0, 1, 501)
records = []

for w1 in weights:
    w2 = 1 - w1
    ret = portfolio_return(w1, r1, r2)
    var = portfolio_variance(w1, sd1, sd2, corr)
    sd = np.sqrt(var)
    esg_score = portfolio_esg(w1, esg1, esg2)
    sharpe = (ret - rf) / sd if sd > 0 else np.nan
    utility = utility_function(ret, var, esg_score, gamma, lam)

    records.append([w1, w2, ret, sd, var, esg_score, sharpe, utility])

df = pd.DataFrame(records, columns=[
    "Weight_A", "Weight_B", "Return", "Risk", "Variance",
    "ESG", "Sharpe", "Utility"
])

best = df.loc[df["Utility"].idxmax()]

print("\nINVESTOR PROFILE")
print(f"Risk style: {risk_profile}")
print(f"Sustainability style: {esg_profile}")

print("\nRECOMMENDED ALLOCATION")
print(f"Asset A: {best['Weight_A']*100:.2f}%")
print(f"Asset B: {best['Weight_B']*100:.2f}%")

print("\nPORTFOLIO CHARACTERISTICS")
print(f"Expected return: {best['Return']*100:.2f}%")
print(f"Portfolio risk: {best['Risk']*100:.2f}%")
print(f"Portfolio ESG score: {best['ESG']:.2f}")
print(f"Sharpe ratio: {best['Sharpe']:.3f}")
print(f"Utility: {best['Utility']:.4f}")

summary = pd.DataFrame({
    "Asset": ["Asset A", "Asset B"],
    "Recommended Weight": [best["Weight_A"], best["Weight_B"]]
})

print("\nTABLE OF RECOMMENDED WEIGHTS")
print(summary.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(df["Risk"], df["Return"], label="ESG-efficient frontier")
plt.scatter(best["Risk"], best["Return"], s=120, marker="D", label="Recommended portfolio")
plt.annotate(
    "Recommended mix",
    (best["Risk"], best["Return"]),
    textcoords="offset points",
    xytext=(10, 8)
)
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Expected Return")
plt.title("Sustainable Portfolio Recommendation")
plt.grid(True)
plt.legend()
plt.show()