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
import numpy as np

# Path for files (requires it to be downloaded locally)
# California, 2007-2017
path = "/Users/jananidhileepan/Desktop/Don't. Even./University/Imperial College London/Year 2/Dissertation/GitHub/fairness_smoke_alarm/data/raw/hmda/"
os.chdir(path)

'''
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
'''

# Clean data
df_merge = pd.read_csv("concat.csv")
df = df_merge.sort_values('as_of_year', kind='stable') # Reorganise year
df = df.iloc[:, 2:].reset_index(drop=True) # Delete unnamed columns

# Drop instances where action_taken == 4 - 8
# 6 is when the loan is purchased by an institution
# This has no information on individuals and is therefore not useful for us
# 7 is where the loan was denied at preapproval but we have no information what happens next
# 4, 5, 8 are where the application is incomplete and is therefore not useful

print("The percentage of entries where the loan was purchased by an institution is", round(len(df[df['action_taken'] == 6])/len(df),2))
print("The percentage of applications that are denied at preapproval is", round(len(df[df['action_taken'] == 7])/len(df),2))
print("The percentage of applications that are incomplete is",round(len(df[df['action_taken'] == 4] + df[df['action_taken'] == 5] + df[df['action_taken'] == 8])/len(df),2))

df = df[df.action_taken != 6]
df = df[df.action_taken != 7]
df = df[(df.action_taken != 4) & (df.action_taken != 5) & (df.action_taken != 8)]
df.groupby("action_taken").size()

# Drop all rows where loan_type == 4 (FSA/RHS loans, as they have a different standard and are negligible in the dataset)
df = df[df.loan_type != 4]

# ------------------------------------------------------------------------------
# 01. Log missing values
# ------------------------------------------------------------------------------

# For each variable there are the equivalents of "nan"
# e.g. Ethnicity has "not applicable"
# For each such variable, replace with nan where appropriate

# 1. Applicant race
# 6 = "Information not provided by applicant in mail, Internet, or telephone application"
# 7 = "Not applicable"
df.loc[df['applicant_race_1'] > 5, 'applicant_race_1'] = np.nan

# 2. Applicant sex
# 3 = "Information not provided by applicant in mail, Internet, or telephone application"
# 4 = "Not applicable"
df.loc[df['applicant_sex'] > 2, 'applicant_sex'] = np.nan

# 3. Applicant ethnicity
# 3 = "Information not provided by applicant in mail, Internet, or telephone application"
# 4 = "Not applicable"
df.loc[df['applicant_ethnicity'] > 2, 'applicant_ethnicity'] = np.nan

# Check missingness by column
missing_df = df.isnull().sum(axis=0)
print(missing_df)

# ------------------------------------------------------------------------------
# 02. Diagnose and treat missing values for race, sex, ethnicity
# ------------------------------------------------------------------------------

# 1. Applicant race
print("The proportion of missing values in the race column is", round(missing_df["applicant_race_1"]/len(df),2))

# By year
df.groupby("as_of_year")["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By sex
df.groupby("applicant_sex")["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By ethnicity - ethnicity = 1 (Hispanic/Latino) has much more missingness
df.groupby("applicant_ethnicity")["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By action taken
df.groupby("action_taken")["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By income
income_bins = pd.cut( # Bin into 50,000 as buckets
    df["applicant_income_000s"],
    bins=range(0, int(df["applicant_income_000s"].max()) + 50, 50)
)

df.groupby(income_bins)["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By loan amount - looks like outlier loan amounts
loan_bins = pd.cut( # Bin into 50,000 as buckets
    df["loan_amount_000s"],
    bins=range(0, int(df["loan_amount_000s"].max()) + 50, 50)
)

df.groupby(loan_bins)["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By loan type
df.groupby("loan_type")["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By lien status - not securied by a lien (3) has much higher missingness
df.groupby("lien_status")["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By tract to msamd income - drops significantly as relative income increases
msamd_income = pd.cut( # Bin into 5% as buckets
    df["tract_to_msamd_income"],
    bins=range(0, int(df["tract_to_msamd_income"].max()) + 5, 5)
)

df.groupby(msamd_income)["applicant_race_1"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# Assume MAR.
# For sensitive variables such as race, sex, and ethnicity, I will not do multiple imputation
# It is morally questionable and statistically risky given multicollinearity between race and ethnicity

# Convert nan values to 0 for race, sex, ethnicity for modelling
df['applicant_race_1'] = df['applicant_race_1'].fillna(0)
df['applicant_sex'] = df['applicant_sex'].fillna(0)
df['applicant_ethnicity'] = df['applicant_ethnicity'].fillna(0)

# ------------------------------------------------------------------------------
# 03. Diagnose and treat missing values for income, loan amount, tract to ms/amd
# ------------------------------------------------------------------------------

# 1. Loan amount
print("The proportion of data with missing loan amounts is", round(missing_df['loan_amount_000s']/len(df),2))
df = df[df['loan_amount_000s'].notna()] # Not even 0.001% dropped

# Filter and drop rows that have basically no information across the row
filtered = df[
    (df["applicant_race_1"] == 0) &
    (df["applicant_sex"] == 0) &
    (df["applicant_ethnicity"] == 0) &
    (df["applicant_income_000s"].isna()) &
    (df["tract_to_msamd_income"].isna())
]

print(len(filtered)) # 1420 rows, or 0.04% of the dataset

df = df[
    ~(
        (df["applicant_race_1"] == 0) &
        (df["applicant_sex"] == 0) &
        (df["applicant_ethnicity"] == 0) &
        (df["applicant_income_000s"].isna()) &
        (df["loan_amount_000s"].isna()) &
        (df["tract_to_msamd_income"].isna())
    )
]

# Filter and drop rows that have at least two continuous variable entries missing, as there is no multiple imputation that can be done.

continuous_vars = [
    "applicant_income_000s",
    "loan_amount_000s",
    "tract_to_msamd_income"
]

df[continuous_vars].isna().sum(axis=1).value_counts().sort_index()

df = df[
    df[continuous_vars].isna().sum(axis=1) < 2 # Drop 3799 entries
]

# 2. Tract to MSA/MD income
# Where it is missing, it is because it is outside county boundaries. I cannot solve this missing data and must drop it.
# Since it is a small percentage, it should not skew the results too strongly.

df = df[df['tract_to_msamd_income'].notna()] # 0.2% dropped

# 3. Applicant income
print("The proportion of missing values in the income column is", round(missing_df["applicant_income_000s"]/len(df),2))

# By year
df.groupby("as_of_year")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By sex - predominantly the missing sex entries also have missing income
df.groupby("applicant_sex")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By ethnicity - predominantly the missing ethnicity entries also have missing income
df.groupby("applicant_ethnicity")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By action taken
df.groupby("action_taken")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By race - predominantly the missing race entries also have missing income
df.groupby("applicant_race_1")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By loan amount - extremely high loan amounts have very high missing values
loan_bins = pd.cut(
    df["loan_amount_000s"],
    bins=[
        0, 50, 100, 150, 200, 250, 300, 400, 500,
        750, 1000, 1500, 2500, float("inf")
    ]
)

df.groupby(loan_bins)["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By loan type - VA-guaranteed loans (2) have much more missingness
df.groupby("loan_type")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# By lien status
df.groupby("lien_status")["applicant_income_000s"].agg(
    missing_count=lambda x: x.isna().sum(),
    total_count="size",
    missing_pct=lambda x: x.isna().mean() * 100
)

# Check how many rows left of missing income
print(df["applicant_income_000s"].isna().sum()/len(df), "of income is missing.") # 6%

# Check how many are associated with missing entries for race, sex, and ethnicity
income_missing = df["applicant_income_000s"].isna()

all_demo_missing = (
    (df["applicant_race_1"] == 0) &
    (df["applicant_ethnicity"] == 0) &
    (df["applicant_sex"] == 0)
)

100 * (income_missing & all_demo_missing).sum() / income_missing.sum()

# As a lot of this corresponds with missing values for race/sex/ethnicity
# And the threshold to drop instead of imputation is 5%, I just drop
# 5-10% is a judgement call

# Drop rows with missing income
df = df[df['applicant_income_000s'].notna()] # 0.2% dropped

# Verify no missingness by column
missing_df_updated = df.isnull().sum(axis=0)
print(missing_df_updated)

# ------------------------------------------------------------------------------
# 04. Clean data and save
# ------------------------------------------------------------------------------

# Rename columns
columns = ['year', 'action_taken', 'race', 'sex', 'ethnicity', 'income_000s',
            'loan_amount_000s', 'loan_type', 'lien_status', 'tract_to_msamd_income']

df.columns = columns

# Save
os.chdir("../../")
df.to_csv("processed/cleaned_missing.csv")