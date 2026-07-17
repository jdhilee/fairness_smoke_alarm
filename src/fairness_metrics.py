# fairness_metrics.py
import pandas as pd
import numpy as np

dict_references = {"race": 4, "sex": 0, "ethnicity": 0}

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

def get_bin_edges(hmda_full, B=10):
    _, edges = pd.qcut(hmda_full[hmda_full['year'] == 2007]['probabilities'], q=B, retbins=True, duplicates='drop')
    return edges