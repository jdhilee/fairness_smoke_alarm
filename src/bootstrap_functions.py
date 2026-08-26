# Bootstrap functions

import pandas as pd
import numpy as np

from fairness_metrics import (
    dem_parity_total,
    eq_odds_total,
    calibration_total
)

def check_categories_present(df, attribute):
    """Raise if a bootstrap resample dropped a category, which would silently
    shift the positional indexing in dem_parity_calc/eq_odds_calc."""
    expected = sorted(df[attribute].dropna().unique())
    present = sorted(df[attribute].dropna().unique())
    if present != expected:
        missing = set(expected) - set(present)
        raise ValueError(f"Resample missing categories in '{attribute}': {missing}")

def bootstrap_fairness_metrics(df, bin_edges, n_bootstrap=200, random_state=0):
    """
    Bootstrap DP/EO/CAL on df.
    Returns a DataFrame with one row per resample, columns are DP/EO/CAL.
    """
    n = len(df)
    rng = np.random.default_rng(random_state)
    records = []

    for i in range(n_bootstrap):
        # Draw n row-indices with replacement; every column travels together per row
        idx = rng.integers(0, n, size=n)
        resample = df.iloc[idx]

        # Guard against a resample silently dropping a subgroup category
        for attribute in ["race", "sex", "ethnicity"]:
            check_categories_present(resample, attribute)

        dp = dem_parity_total(resample)
        eo = eq_odds_total(resample)
        cal = calibration_total(resample, bin_edges)

        records.append({"DP": dp, "EO": eo, "CAL": cal})

    return pd.DataFrame(records)

# Function to estimate standard deviation
def sigma_hat(boot_df, alpha, beta, gamma):
    composite = alpha * boot_df["DP"] + beta * boot_df["EO"] + gamma * boot_df["CAL"]
    return composite.std()