# Regulatory Vagueness and Fairness Drift: A Statistical Critique of Post-Deployment AI Monitoring

Code accompanying the MSc dissertation (Imperial College London, MLDS, 2026). Implements composite fairness monitoring (CUSUM / EWMA-residual) on frozen HMDA mortgage models, and a Monte Carlo sweep over composite-metric weightings to characterise the miss-rate simplex referenced in Article 72 of the EU AI Act.

## Repository structure

```
data/                                       # Empty on GitHub (too large) — see data/README.md for replication steps
milestones/                                 # Milestone 1 and Milestone 2 submissions
notebooks/
├── exploratory/
│   └── preliminary_checks_hmda.ipynb       # Early check of whether HMDA was fit for purpose
└── analysis/
    ├── 01_clean_hmda_2007_2017.ipynb       # Cleaning and preprocessing, 2007–2017
    ├── 02_eda_hmda_2007_2017.ipynb         # Exploratory data analysis
    ├── 03_logreg_hmda_2007.ipynb           # Frozen logistic regression trained on 2007 data
    ├── 04_fairness_calculations.ipynb      # DP / EO / CAL by year -> df_fairness
    ├── 05_changepoint_detection.ipynb      # CUSUM / EWMA / AND / OR detectors on df_fairness
    └── 06_monte_carlo_simulations.ipynb    # 2,000-draw Dirichlet sweep over (alpha, beta, gamma) -> df_mc
src/
├── bootstraps_functions.py                 # Bootstrap SD estimation for DP/EO/CAL noise terms
├── changepoint_detection_functions.py      # CUSUM and EWMA-residual detector implementations
├── fairness_metrics.py                     # DP / EO / CAL computation, composite scoring
└── monte_carlo_simulation_functions.py     # Dirichlet sampling and miss-rate evaluation against OR-rule referee
```

## Data

California HMDA mortgage records, 2007–2017. Not included in this repository due to size — see `data/README.md` for replication instructions (source, filtering, expected directory layout).

## Run order

1. `notebooks/exploratory/preliminary_checks_hmda.ipynb` — optional; documents the initial suitability check of the dataset, not required to reproduce results.
2. `notebooks/analysis/01_clean_hmda_2007_2017.ipynb` — cleans and preprocesses the raw HMDA extract.
3. `notebooks/analysis/02_eda_hmda_2007_2017.ipynb` — exploratory analysis of the cleaned data.
4. `notebooks/analysis/03_logreg_hmda_2007.ipynb` — trains the frozen 2007 logistic regression model.
5. `notebooks/analysis/04_fairness_calculations.ipynb` — computes per-year DP/EO/CAL against the frozen model, produces `df_fairness`.
6. `notebooks/analysis/05_changepoint_detection.ipynb` — runs the four detector variants (CUSUM, EWMA, AND, OR) against the fixed 2007 baseline.
7. `notebooks/analysis/06_monte_carlo_simulations.ipynb` — draws 2,000 (α, β, γ) weightings from Dirichlet(1,1,1), evaluates each against the OR-rule referee, produces `df_mc` and the ternary miss-rate surface.

Each analysis notebook depends on the output of the previous one; run in numeric order.

## Note on scope

This code supports the empirical Results chapter.