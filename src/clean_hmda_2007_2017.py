# ------------------------------------------------------------------------------
# Name:        clean_hmda_2007_2017.py
# Purpose:     Merge and clean HMDA data from 2007-2017
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 00. Setup
# ------------------------------------------------------------------------------

# Import necessary libraries
import pandas as pd
import os

# Path for files (requires it to be downloaded locally)
# California, 2007-2017
path = "/Users/jananidhileepan/Desktop/Don't. Even./University/Imperial College London/Year 2/Dissertation/GitHub/fairness_smoke_alarm/data/raw/hmda/"
os.chdir(path)

# Reduce CSV size
cols_to_keep = ["as_of_year", "action_taken", "applicant_race_1", "applicant_sex", 
                "applicant_ethnicity", "applicant_income_000s", "loan_amount_000s", 
                "loan_type", "lien_status", "tract_to_msamd_income"]

for filename in os.listdir(path):
    if filename.endswith(".csv"):
        df_raw = pd.read_csv(filename)
        df = df_raw[cols_to_keep] # Only keep needed columns
        df.to_csv(filename)

# Concatenate
flist = []
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        flist.append(df)

df_merge = pd.concat(flist, axis=0, ignore_index=True)
df_merge.drop(df_merge.columns[0], axis=1) # Drop index column
df_merge.to_csv("concat.csv")