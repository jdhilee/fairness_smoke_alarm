import pandas as pd
import numpy as np

# Function to estimate standard deviation
def sigma_hat(boot_df, alpha, beta, gamma):
    composite = alpha * boot_df["DP"] + beta * boot_df["EO"] + gamma * boot_df["CAL"]
    return composite.std()

# Class that defines the CUSUM alarm

class CUSUMDetector:
    def __init__(self, series, sigma_hat, k=0.5, h=4.77, baseline_year=2007):
        """
        series: pandas Series indexed by year, including baseline_year
        sigma_hat: standard deviation to standardize with
        onset year: last good year before the alarm rang
        """
        self.series = series
        self.baseline_year = baseline_year
        self.sigma_hat = sigma_hat
        self.k = k
        self.h = h

        self.mu_0 = self.series.loc[baseline_year]
        self.years = self.series.index.drop(baseline_year)

        # Populated by run()
        self.alarm = None
        self.onset_year = None
        self.C_plus = None

    def run(self):
        alarm = pd.Series(0, index=self.years)
        onset_year = pd.Series(None, index=self.years, dtype="object")
        C_plus_trajectory = pd.Series(0.0, index=self.years)

        C_plus_t = 0
        N_plus = 0

        for year in self.years:
            x_t = self.series.loc[year]
            y_t = (x_t - self.mu_0) / self.sigma_hat
            C_plus_t = max(0, y_t - self.k + C_plus_t)
            C_plus_trajectory.loc[year] = C_plus_t

            N_plus = 0 if C_plus_t == 0 else N_plus + 1

            if C_plus_t > self.h:
                alarm.loc[year] = 1
                onset_year.loc[year] = year - N_plus

        self.alarm = alarm
        self.onset_year = onset_year
        self.C_plus = C_plus_trajectory
        return alarm, onset_year, C_plus_trajectory

# Function that defines the "referee", i.e. the ground-truth proxy for fairness

def referee_cusum(df_fairness, boot_df, k=0.5, h=4.77, baseline_year=2007):
    """
    Runs CUSUM independently on DP, EO, and CAL (each with its own mu_0 and
    sigma_hat), then combines via OR: referee alarms in year t if ANY of the
    three individual metrics alarm in year t.
    """
    detectors = {}

    for metric in ["DP", "EO", "CAL"]:
        series = df_fairness[metric]
        se = boot_df[metric].std()  # marginal SE for this metric alone

        det = CUSUMDetector(series, sigma_hat=se, k=k, h=h, baseline_year=baseline_year)
        det.run()
        detectors[metric] = det

    # OR combination: referee alarms if any individual detector alarms
    alarm_matrix = pd.DataFrame({
        metric: detectors[metric].alarm for metric in detectors
    })
    referee_alarm = (alarm_matrix.sum(axis=1) > 0).astype(int)

    return referee_alarm, detectors

# EWMA Alarm

class EWMAResidualDetector:
    def __init__(self, series, sigma_hat, lam=0.4, L=3, baseline_year=2007):
        """
        series: pandas Series indexed by year, including baseline_year
        sigma_hat: bootstrap SD of the composite metric (the same one CUSUM uses)
        lam: EWMA smoothing parameter in (0, 1); weight on the newest year
        L: control-limit multiplier (L=3 -> ARL0 ~ 370, matching the CUSUM)
        """
        self.series = series
        self.baseline_year = baseline_year
        self.sigma_hat = sigma_hat
        self.lam = lam
        self.L = L

        # The 2007 value is the forecast anchor (z_0): the first year is predicted
        # from baseline, then the forecast adapts as it walks forward.
        self.z0 = self.series.loc[baseline_year]
        self.years = self.series.index.drop(baseline_year)

        # Prediction errors vary more than the metric, so scale sigma_hat up.
        # sigma_e = sigma_hat * sqrt(2 / (2 - lam)).
        self.sigma_e = self.sigma_hat * np.sqrt(2 / (2 - self.lam))

        # Populated by run()
        self.alarm = None
        self.residuals = None
        self.forecast = None

    def run(self):
        alarm = pd.Series(0, index=self.years)
        residuals = pd.Series(0.0, index=self.years)
        forecast = pd.Series(0.0, index=self.years)

        z_prev = self.z0  # forecast for the first monitored year is the 2007 anchor

        for year in self.years:
            x_t = self.series.loc[year]

            # one-step residual: actual minus what the smoothed trend predicted
            e_t = x_t - z_prev
            residuals.loc[year] = e_t
            forecast.loc[year] = z_prev

            # one-sided upper alarm: metric running above its own recent trend
            if e_t > self.L * self.sigma_e:
                alarm.loc[year] = 1

            # update the smoothed forecast for next year
            z_prev = self.lam * x_t + (1 - self.lam) * z_prev

        self.alarm = alarm
        self.residuals = residuals
        self.forecast = forecast
        return alarm, residuals, forecast

# Function that defines the "referee", i.e. the ground-truth proxy for fairness

def referee_ewma(df_fairness, boot_df):
    """
    Runs EWMA independently on DP, EO, and CAL,
    then combines via OR: referee alarms in year t if ANY of the
    three individual metrics alarm in year t.
    """
    detectors = {}

    for metric in ["DP", "EO", "CAL"]:
        series = df_fairness[metric]
        se = boot_df[metric].std()  # marginal SE for this metric alone

        ewma_alarm = EWMAResidualDetector(series, sigma_hat=se)
        ewma_alarm.run()
        detectors[metric] = ewma_alarm

    # OR combination: referee alarms if any individual detector alarms
    alarm_matrix = pd.DataFrame({
        metric: detectors[metric].alarm for metric in detectors
    })
    referee_alarm = (alarm_matrix.sum(axis=1) > 0).astype(int)

    return referee_alarm, detectors


# AND/OR Gate for Alarm

def and_or_gate(cusum, ewma, gate="and", return_df = False):
    comparison_alarms = pd.DataFrame({
        "cusum_alarm": cusum.alarm,
        "ewma_alarm": ewma.alarm
    })

    comparison_alarms["and_alarm"] = (comparison_alarms["cusum_alarm"] & comparison_alarms["ewma_alarm"]).astype(int)
    comparison_alarms["or_alarm"] = (comparison_alarms["cusum_alarm"] | comparison_alarms["ewma_alarm"]).astype(int)

    if return_df:
        return comparison_alarms

    if gate == "and": # AND gate
        return comparison_alarms["and_alarm"]
    elif gate == "or": # OR gate
        return comparison_alarms["or_alarm"]
    else:
        raise ValueError("Gate must be 'and' or 'or'.")

# Referee AND/OR Gates

def and_or_gate_referee(detectors_cusum, detectors_ewma, gate="and", return_df=False):
    """
    Per-metric gating: for each of DP/EO/CAL, combine that metric's CUSUM and
    EWMA detectors with the gate, then OR across the three metrics.
    """
    and_cols = {}
    or_cols = {}

    for metric in ["DP", "EO", "CAL"]:
        gated = and_or_gate(
            detectors_cusum[metric],
            detectors_ewma[metric],
            return_df=True
        )
        and_cols[metric] = gated["and_alarm"]
        or_cols[metric] = gated["or_alarm"]

    and_df = pd.DataFrame(and_cols)
    or_df = pd.DataFrame(or_cols)

    and_referee = (and_df.sum(axis=1) > 0).astype(int)
    or_referee = (or_df.sum(axis=1) > 0).astype(int)

    if return_df:
        out = and_df.add_suffix("_and").join(or_df.add_suffix("_or"))
        out["and_referee"] = and_referee
        out["or_referee"] = or_referee
        return out

    if gate == "and":
        return and_referee
    elif gate == "or":
        return or_referee
    else:
        raise ValueError("Gate must be 'and' or 'or'.")

# Miss rate

def miss_rate(alarm, referee, **extra):
    misses = (referee == 1) & (alarm == 0)
    n_flagged = (referee == 1).sum()

    if n_flagged == 0:
        rate = np.nan
    else:
        rate = misses.sum() / n_flagged

    fired = alarm[alarm == 1]

    return {
        "miss_rate": rate,
        "any_miss": bool(misses.any()),
        "ever_alarmed": bool(len(fired) > 0),
        "n_transitions": int((alarm.diff().fillna(0) != 0).sum()),
        "first_alarm_year": int(fired.index[0]) if len(fired) > 0 else np.nan,
        "referee_flagged_years": int(n_flagged),
        **extra,
    }