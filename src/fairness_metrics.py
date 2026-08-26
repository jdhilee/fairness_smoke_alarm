# fairness_metrics.py
import pandas as pd
import numpy as np

dict_references = {"race": 4, "sex": 0, "ethnicity": 0}

# DP, EO, and CAL

def dem_parity_calc(df, attribute):
    dem_parity = (
        df
        .groupby(attribute)['predictions']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(2)
    )
    dem_parity_values = []
    for i in range(len(dem_parity)):
        if i != dict_references[attribute]:
            dem_parity_values.append(abs(dem_parity.iloc[i][1] - dem_parity.iloc[dict_references[attribute]][1]))
    return sum(dem_parity_values) / (len(dem_parity) - 1)

def dem_parity_total(df):
    race_dem_parity = dem_parity_calc(df, attribute="race")
    sex_dem_parity = dem_parity_calc(df, attribute="sex")
    ethnicity_dem_parity = dem_parity_calc(df, attribute="ethnicity")
    return (race_dem_parity + sex_dem_parity + ethnicity_dem_parity) / len(dict_references)

def eq_odds_calc(df, attribute):
    ref = dict_references[attribute]
    tpr = (
        df[df['action_taken'] == 1]
        .groupby(attribute)['predictions']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(2)
    )
    fpr = (
        df[df['action_taken'] == 0]
        .groupby(attribute)['predictions']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(2)
    )
    eo_values = []
    for i in range(len(tpr)):
        if i != ref:
            tpr_gap = abs(tpr.iloc[i][1] - tpr.iloc[ref][1])
            fpr_gap = abs(fpr.iloc[i][1] - fpr.iloc[ref][1])
            eo_values.append(0.5 * (tpr_gap + fpr_gap))
    return sum(eo_values) / (len(tpr) - 1)

def eq_odds_total(df):
    race = eq_odds_calc(df, "race")
    sex = eq_odds_calc(df, "sex")
    ethnicity = eq_odds_calc(df, "ethnicity")
    return (race + sex + ethnicity) / len(dict_references)

def calibration_calc(df, attribute, bin_edges):
    df = df.copy()
    B = len(bin_edges) - 1
    df['prob_bin'] = pd.cut(df['probabilities'], bins=bin_edges, labels=False, include_lowest=True)
    cal_values = []
    subgroups = sorted(df[attribute].dropna().unique())
    for group in subgroups:
        bin_errors = []
        for b in range(B):
            grp_bin = df[(df[attribute] == group) & (df['prob_bin'] == b)]
            if len(grp_bin) > 0:
                avg_pred_prob = grp_bin['probabilities'].mean()
                actual_rate = grp_bin['action_taken'].mean()
                bin_errors.append(abs(avg_pred_prob - actual_rate))
            else:
                bin_errors.append(0)
        cal_values.append(sum(bin_errors) / B)
    return sum(cal_values) / len(subgroups)

def calibration_total(df, bin_edges):
    race = calibration_calc(df, "race", bin_edges)
    sex = calibration_calc(df, "sex", bin_edges)
    ethnicity = calibration_calc(df, "ethnicity", bin_edges)
    return (race + sex + ethnicity) / len(dict_references)

# Bin edges for CAL

def get_bin_edges(hmda_full, B=10):
    _, edges = pd.qcut(hmda_full[hmda_full['year'] == 2007]['probabilities'], q=B, retbins=True, duplicates='drop')
    return edges

def unfairness_score(df_fairness, alpha=1/3, beta=1/3, gamma=1/3):
    return alpha * df_fairness['DP'] + beta * df_fairness['EO'] + gamma * df_fairness['CAL']

# Confidence Intervals

def dem_parity_calc_with_ci(df, attribute, z=1.96):
    dem_parity = (
        df
        .groupby(attribute)['predictions']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(4)
    )
    
    # Group sizes (needed for standard errors)
    group_counts = df.groupby(attribute)['predictions'].count()
    
    gaps = []
    lower_bounds = []
    upper_bounds = []
    
    ref = dict_references[attribute]
    p_ref = dem_parity.iloc[ref][1]
    n_ref = group_counts.iloc[ref]
    
    for i in range(len(dem_parity)):
        if i != ref:
            p_g = dem_parity.iloc[i][1]
            n_g = group_counts.iloc[i]
            
            gap = abs(p_g - p_ref)
            
            # Standard error of the difference between two proportions
            se = np.sqrt((p_ref * (1 - p_ref)) / n_ref + (p_g * (1 - p_g)) / n_g)
            
            gaps.append(gap)
            lower_bounds.append(max(0, gap - z * se))
            upper_bounds.append(min(1, gap + z * se))
    
    n_groups = len(dem_parity) - 1
    return (
        sum(gaps) / n_groups,
        sum(lower_bounds) / n_groups,
        sum(upper_bounds) / n_groups
    )


def dem_parity_total_with_ci(df):
    race_dp, race_lo, race_hi = dem_parity_calc_with_ci(df, "race")
    sex_dp, sex_lo, sex_hi = dem_parity_calc_with_ci(df, "sex")
    eth_dp, eth_lo, eth_hi = dem_parity_calc_with_ci(df, "ethnicity")
    
    A = len(dict_references)
    return (
        (race_dp + sex_dp + eth_dp) / A,
        (race_lo + sex_lo + eth_lo) / A,
        (race_hi + sex_hi + eth_hi) / A
    )

def eq_odds_calc_with_ci(df, attribute, z=1.96):
    ref = dict_references[attribute]
    
    # TPR: among actually approved (y=1)
    df_approved = df[df['action_taken'] == 1]
    tpr = (
        df_approved
        .groupby(attribute)['predictions']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(4)
    )
    tpr_counts = df_approved.groupby(attribute)['predictions'].count()
    
    # FPR: among actually denied (y=0)
    df_denied = df[df['action_taken'] == 0]
    fpr = (
        df_denied
        .groupby(attribute)['predictions']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(4)
    )
    fpr_counts = df_denied.groupby(attribute)['predictions'].count()
    
    eo_values, lower_bounds, upper_bounds = [], [], []
    
    for i in range(len(tpr)):
        if i != ref:
            # TPR gap and its SE
            p_tpr_g = tpr.iloc[i][1]
            p_tpr_ref = tpr.iloc[ref][1]
            n_tpr_g = tpr_counts.iloc[i]
            n_tpr_ref = tpr_counts.iloc[ref]
            tpr_gap = abs(p_tpr_g - p_tpr_ref)
            se_tpr = np.sqrt(
                (p_tpr_ref * (1 - p_tpr_ref)) / n_tpr_ref +
                (p_tpr_g * (1 - p_tpr_g)) / n_tpr_g
            )
            
            # FPR gap and its SE
            p_fpr_g = fpr.iloc[i][1]
            p_fpr_ref = fpr.iloc[ref][1]
            n_fpr_g = fpr_counts.iloc[i]
            n_fpr_ref = fpr_counts.iloc[ref]
            fpr_gap = abs(p_fpr_g - p_fpr_ref)
            se_fpr = np.sqrt(
                (p_fpr_ref * (1 - p_fpr_ref)) / n_fpr_ref +
                (p_fpr_g * (1 - p_fpr_g)) / n_fpr_g
            )
            
            # EO for this subgroup is average of TPR and FPR gaps
            eo_gap = 0.5 * (tpr_gap + fpr_gap)
            eo_lo = 0.5 * (max(0, tpr_gap - z * se_tpr) + max(0, fpr_gap - z * se_fpr))
            eo_hi = 0.5 * (min(1, tpr_gap + z * se_tpr) + min(1, fpr_gap + z * se_fpr))
            
            eo_values.append(eo_gap)
            lower_bounds.append(eo_lo)
            upper_bounds.append(eo_hi)
    
    n_groups = len(tpr) - 1
    return (
        sum(eo_values) / n_groups,
        sum(lower_bounds) / n_groups,
        sum(upper_bounds) / n_groups
    )


def eq_odds_total_with_ci(df):
    race_eo, race_lo, race_hi = eq_odds_calc_with_ci(df, "race")
    sex_eo, sex_lo, sex_hi = eq_odds_calc_with_ci(df, "sex")
    eth_eo, eth_lo, eth_hi = eq_odds_calc_with_ci(df, "ethnicity")
    
    A = len(dict_references)
    return (
        (race_eo + sex_eo + eth_eo) / A,
        (race_lo + sex_lo + eth_lo) / A,
        (race_hi + sex_hi + eth_hi) / A
    )

def calibration_calc_with_ci(df, attribute, bin_edges, z=1.96):
    df = df.copy()
    B = len(bin_edges) - 1  # derived, can't drift out of sync
    df["prob_bin"] = pd.cut(df["probabilities"], bins=bin_edges, labels=False, include_lowest=True)

    subgroups = sorted(df[attribute].dropna().unique())
    cal_values, lower_bounds, upper_bounds = [], [], []

    for group in subgroups:
        bin_errors, bin_los, bin_his = [], [], []

        for b in range(B):
            grp_bin = df[(df[attribute] == group) & (df["prob_bin"] == b)]
            n = len(grp_bin)

            if n > 0:
                avg_pred_prob = grp_bin["probabilities"].mean()
                actual_rate = grp_bin["action_taken"].mean()
                error = abs(avg_pred_prob - actual_rate)
                se = np.sqrt((actual_rate * (1 - actual_rate)) / n)
                lo = max(0, error - z * se)
                hi = min(1, error + z * se)
            else:
                error, lo, hi = 0, 0, 0

            bin_errors.append(error)
            bin_los.append(lo)
            bin_his.append(hi)

        cal_values.append(sum(bin_errors) / B)
        lower_bounds.append(sum(bin_los) / B)
        upper_bounds.append(sum(bin_his) / B)

    return (
        sum(cal_values) / len(subgroups),
        sum(lower_bounds) / len(subgroups),
        sum(upper_bounds) / len(subgroups)
    )


def calibration_total_with_ci(df, bin_edges):
    attributes = ["race", "sex", "ethnicity"]
    results = [calibration_calc_with_ci(df, attr, bin_edges) for attr in attributes]
    cal = sum(r[0] for r in results) / len(attributes)
    lo = sum(r[1] for r in results) / len(attributes)
    hi = sum(r[2] for r in results) / len(attributes)
    return cal, lo, hi

# Unfairness Score

def unfairness_score(df_fairness, alpha=1/3, beta=1/3, gamma=1/3):
    return alpha * df_fairness['DP'] + beta * df_fairness['EO'] + gamma * df_fairness['CAL']