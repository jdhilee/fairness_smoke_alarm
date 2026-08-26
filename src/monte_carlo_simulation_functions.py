# Monte Carlo simulation functions

import numpy as np
import pandas as pd
from fairness_metrics import unfairness_score
from changepoint_detection_functions import *

# Variance
def sigma_hat(w, Sigma):
    return np.sqrt(w @ Sigma @ w)

# Evaluate the miss rates for the four alarms for a given weighting

def evaluate_weighting(alpha, beta, gamma, df_fairness, Sigma,
                       referee_cusum_alarm, referee_ewma_alarm,
                       referee_and, referee_or,
                       k=0.5, h=4.77, baseline_year=2007):

    composite = unfairness_score(df_fairness, alpha, beta, gamma) # weights passed
    w = np.array([alpha, beta, gamma]) # order: DP, EO, CAL
    se = sigma_hat(w, Sigma)

    cus = CUSUMDetector(composite, sigma_hat=se, k=k, h=h, baseline_year=baseline_year)
    cus.run()

    ewm = EWMAResidualDetector(composite, sigma_hat=se, baseline_year=baseline_year)
    ewm.run()

    and_alarm = and_or_gate(cus, ewm, gate="and")
    or_alarm  = and_or_gate(cus, ewm, gate="or")

    weights = {"alpha": alpha, "beta": beta, "gamma": gamma, "sigma_hat": se}

    return [
        miss_rate(cus.alarm,  referee_cusum_alarm, variant="cusum", **weights),
        miss_rate(ewm.alarm,  referee_ewma_alarm,  variant="ewma",  **weights),
        miss_rate(and_alarm,  referee_and,         variant="and",   **weights),
        miss_rate(or_alarm,   referee_or,          variant="or",    **weights),
    ]