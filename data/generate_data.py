"""
Synthetic Credit Risk Dataset Generator.

Generates a realistic-looking dataset of loan applicants with engineered
correlations that make the classification problem non-trivial.
"""

import os
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES = 5000
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_credit_risk_dataset(
    n_samples: int = N_SAMPLES,
    random_seed: int = RANDOM_SEED,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Generate a synthetic credit risk dataset and save it to CSV.

    Features
    --------
    age                  : Applicant age (22–70)
    income               : Annual income in USD (20k–150k)
    credit_score         : FICO-like score (300–850)
    loan_amount          : Requested loan amount (5k–100k)
    loan_tenure_months   : Loan duration (12–84 months)
    debt_to_income_ratio : Existing debt / annual income (0.0–0.9)
    num_open_accounts    : Number of open credit accounts (1–15)
    num_credit_inquiries : Hard credit inquiries in last 12 months (0–10)
    months_employed      : Months at current employer (0–360)
    has_mortgage         : Binary flag — owns a mortgage (0 or 1)

    Target
    ------
    default : 1 if applicant defaulted, 0 otherwise
    """
    rng = np.random.default_rng(random_seed)

    # ---- Feature generation -----------------------------------------------

    age = rng.integers(22, 71, size=n_samples).astype(float)

    income = rng.lognormal(mean=11.0, sigma=0.5, size=n_samples)
    income = np.clip(income, 20_000, 150_000)

    # Credit score inversely correlated with default risk
    credit_score_base = rng.normal(650, 80, size=n_samples)
    credit_score = np.clip(credit_score_base, 300, 850)

    loan_amount = rng.lognormal(mean=10.5, sigma=0.6, size=n_samples)
    loan_amount = np.clip(loan_amount, 5_000, 100_000)

    loan_tenure_months = rng.choice([12, 24, 36, 48, 60, 72, 84], size=n_samples).astype(float)

    debt_to_income_ratio = rng.beta(2, 5, size=n_samples)          # skewed toward low values

    num_open_accounts = rng.integers(1, 16, size=n_samples).astype(float)

    num_credit_inquiries = rng.integers(0, 11, size=n_samples).astype(float)

    months_employed = rng.exponential(scale=60, size=n_samples)
    months_employed = np.clip(months_employed, 0, 360)

    has_mortgage = rng.binomial(1, 0.35, size=n_samples).astype(float)

    # ---- Default label (engineered correlations) --------------------------
    # Higher log-odds of default for:
    #   low credit score, high DTI, many inquiries, high loan-to-income ratio

    log_odds = (
        -0.02 * age                                         # older → lower risk
        - 0.005 * (income / 1_000)                         # higher income → lower risk
        - 0.008 * credit_score                             # better score → lower risk
        + 0.000_01 * loan_amount                            # larger loan → higher risk
        + 0.003 * loan_tenure_months                        # longer tenure → higher risk
        + 3.5 * debt_to_income_ratio                        # high DTI → higher risk
        - 0.05 * num_open_accounts                          # more accounts → slightly lower risk
        + 0.25 * num_credit_inquiries                       # more inquiries → higher risk
        - 0.003 * months_employed                           # stable job → lower risk
        - 0.3 * has_mortgage                                # mortgage owners → lower risk
        + 2.0                                               # intercept
    )

    prob_default = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(1, prob_default).astype(int)

    # ---- Assemble DataFrame -----------------------------------------------
    df = pd.DataFrame({
        "age": age.round(0).astype(int),
        "income": income.round(2),
        "credit_score": credit_score.round(0).astype(int),
        "loan_amount": loan_amount.round(2),
        "loan_tenure_months": loan_tenure_months.astype(int),
        "debt_to_income_ratio": debt_to_income_ratio.round(4),
        "num_open_accounts": num_open_accounts.astype(int),
        "num_credit_inquiries": num_credit_inquiries.astype(int),
        "months_employed": months_employed.round(0).astype(int),
        "has_mortgage": has_mortgage.astype(int),
        "default": default,
    })

    # ---- Save ----------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    default_rate = df["default"].mean() * 100
    print(f"[DataGen] Generated {len(df):,} samples → {output_path}")
    print(f"[DataGen] Default rate: {default_rate:.1f}%")
    print(f"[DataGen] Feature stats:\n{df.describe().to_string()}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit risk dataset")
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    generate_credit_risk_dataset(
        n_samples=args.samples,
        random_seed=args.seed,
        output_path=args.output,
    )
