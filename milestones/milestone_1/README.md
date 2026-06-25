 # Milestone 1 — Regulatory Vagueness and Fairness Drift: A Statistical Critique of Post-Deployment AI Monitoring

**Author:** Janani Dhileepan  
**CID:** 06026467

---

## Overview

This submission contains the code, data, and report for Milestone 1. The project uses the US Home Mortgage Disclosure Act (HMDA) dataset (California, 2007–2017) to train a logistic regression classifier and construct a composite fairness metric, as a foundation for a post-deployment fairness monitoring framework ("smoke alarm") using change point detection.

---

## File Structure

```
zip/
├── data/
│   └── processed/
│       └── cleaned_missing.csv        # Cleaned dataset used by all notebooks
├── src/
│   ├── clean_hmda_2007_2017.py        # Cleaning script for raw HMDA data
│   ├── eda_hmda_2007_2017.ipynb       # Exploratory data analysis
│   └── analysis_hmda_2007.ipynb       # Logistic regression and fairness metrics
├── milestone_1.ipynb                  # Main milestone notebook
├── milestone_1.pdf                    # PDF version of the milestone notebook
└── README.md
```

---

## How to Run

1. Unzip the folder and preserve the directory structure.
2. Open `milestone_1.ipynb` from the root of the unzipped folder.
3. Run all cells in order. The notebook navigates to `src/` and runs the EDA and analysis notebooks automatically via `%run`.

The notebook reads from `data/processed/cleaned_missing.csv`. The raw HMDA data are not included due to file size; the cleaning logic is documented in `src/clean_hmda_2007_2017.py`.

---

## Dependencies

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
