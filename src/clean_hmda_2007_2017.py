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

# Clean data
# df_merge = pd.read_csv("concat.csv")
df = df_merge.sort_values('as_of_year', kind='stable') # Reorganise year
df = df.iloc[:, 2:].reset_index(drop=True) # Delete unnamed columns

# Drop instances where action_taken == 6
# This is when the loan is purchased by an institution
# This has no information on individuals and is therefore not useful for us

print("The percentage of entries where the loan was purchased by an institution is", round(len(df[df['action_taken'] == 6])/len(df),2))
df = df[df.action_taken != 6] # Remove 18% of entries

# ------------------------------------------------------------------------------
# 01. Missing values
# ------------------------------------------------------------------------------

# Check missingness by column
missing_df = df.isnull().sum(axis=0)
print(missing_df)

df.groupby("action_taken").size()