# Data

This directory contains raw and processed data for two domains used in the fairness smoke alarm analysis. Raw files are stored locally and not committed to this repository.

---

## HMDA (Credit Domain)

### Source
- **Dataset:** Home Mortgage Disclosure Act (HMDA) Loan Application Register
- **Provider:** Consumer Financial Protection Bureau (CFPB)
- **URL:** https://www.consumerfinance.gov/data-research/hmda/historic-data/
- **Years used:** 2007–2017
- **Geography:** California

### Downloading Raw Data
Download state-level LAR files from the URL above and place them in `data/raw/hmda/`. When downloading, select:
- State: California
- Option: **All records** (includes applications, denials, originations — do not use the pre-filtered originated loans file)
- Format: Labels (plain language) — note: switch to HMDA codes only once EDA is complete

Files required:
- `hmda_2007_ca_all-records_labels.csv`
- `hmda_2008_ca_all-records_labels.csv`
- `hmda_2009_ca_all-records_labels.csv`
- `hmda_2010_ca_all-records_labels.csv`
- `hmda_2011_ca_all-records_labels.csv`
- `hmda_2012_ca_all-records_labels.csv`
- `hmda_2013_ca_all-records_labels.csv`
- `hmda_2014_ca_all-records_labels.csv`
- `hmda_2015_ca_all-records_labels.csv`
- `hmda_2016_ca_all-records_labels.csv`
- `hmda_2017_ca_all-records_labels.csv`

### Data Decisions
- **Excluded action_taken == 6** (loan purchased by institution): these are secondary market transactions with no lending decision and systematically missing demographic data
- **Primary applicant only**: fairness metrics are computed on primary applicant demographics; co-applicant variables are retained in the dataset but not used for metric calculation, consistent with fair lending regulatory practice
- **Protected attributes monitored**: race (applicant_race_1) and sex (applicant_sex)
- **Age**: not available in the pre-2018 HMDA schema; noted as a limitation
- **"Information not provided" and "not applicable"** are treated as missing for fairness metric purposes

### Missingness (2007, post-exclusion of purchased loans)
| Variable | Effective Missing Rate |
|----------|----------------------|
| applicant_race_1 | ~21% |
| applicant_ethnicity | ~19% |
| applicant_sex | ~9% |

### Processing
Raw data is processed by `src/data_processing.py`, which:
- Excludes purchased loans (action_taken == 6)
- Selects relevant columns (see variables table below)
- Treats "information not provided" and "not applicable" as missing

### Variables Used
| Variable | Description | Type |
|----------|-------------|------|
| action_taken | Whether loan was originated, denied, withdrawn, etc. | Target |
| action_taken_name | Plain language label for action_taken | Target (labels) |
| applicant_race_1 | Primary applicant race (first selection) | Protected attribute |
| applicant_race_name_1 | Plain language label for applicant_race_1 | Protected attribute (labels) |
| applicant_race_2 to 5 | Additional race selections (HMDA allows multiple) | Protected attribute |
| applicant_sex | Primary applicant sex | Protected attribute |
| applicant_sex_name | Plain language label for applicant_sex | Protected attribute (labels) |
| applicant_ethnicity | Primary applicant ethnicity (Hispanic/Latino) | Protected attribute |
| applicant_ethnicity_name | Plain language label for applicant_ethnicity | Protected attribute (labels) |
| co_applicant_race_1 to 5 | Co-applicant race (retained, not used in metrics) | Reference only |
| co_applicant_sex | Co-applicant sex (retained, not used in metrics) | Reference only |
| co_applicant_ethnicity | Co-applicant ethnicity (retained, not used in metrics) | Reference only |
| applicant_income_000s | Gross annual income in thousands of dollars | Feature |
| loan_amount_000s | Loan amount in thousands of dollars | Feature |
| loan_type | Loan type code (conventional, FHA, VA, FSA/RHS) | Feature |
| loan_type_name | Plain language label for loan_type | Feature (labels) |
| lien_status | Lien status code (first lien, subordinate lien, etc.) | Feature |
| lien_status_name | Plain language label for lien_status | Feature (labels) |
| tract_to_msamd_income | Tract median income as % of MSA/MD median income | Feature |
| county_code | Five-digit FIPS county code | Geographic reference |
| county_name | County name | Geographic reference (labels) |
| denial_reason_1 to 3 | Up to three reasons for denial (only populated for denied applications) | Reference only |
| denial_reason_name_1 to 3 | Plain language labels for denial reasons | Reference only |
| as_of_year | Year of the filing period | Time identifier |

### Processed Data
| File | Description |
|------|-------------|
| `data/processed/hmda_cleaned.csv` | Cleaned and filtered HMDA dataset, all years stacked |

### Time Structure
| Period | Years | Purpose |
|--------|-------|---------|
| Training | 2007–2008 | Model training |
| Pre-deployment validation | 2009–2010 | Fairness baseline establishment |
| Monitoring | 2011–2017 | Change point detection window |