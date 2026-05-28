# Data

## Source
- **Dataset:** Home Mortgage Disclosure Act (HMDA)
- **Provider:** Consumer Financial Protection Bureau (CFPB)
- **URL:** https://ffiec.cfpb.gov/npwui
- **Years used:** 2007-2017

## Raw Data
Raw files are stored locally and not committed to this repository.
To reproduce, filter using the following:
- California
- Mortgages for first lien, owner-occupied, 1-4 family homes
- Plain language and HMDA codes **TODO: only HMDA codes?**

Download the following files from the URL above and place them in `data/raw/hmda/`:
- 2007-2017

**TODO: Update as needed**
## Processing
Raw data is processed by `src/data_processing.py`, which:
- [e.g. filters to mortgage applications in the US]
- [e.g. removes rows with missing protected attribute data]
- [e.g. selects relevant columns]

## Processed Data
The processed dataset used for analysis is stored in `data/processed/`.

| File | Description |
|------|-------------|
| `hmda_cleaned.csv` | Cleaned and filtered HMDA dataset |

## Variables Used
| Variable | Description | Type |
|----------|-------------|------|
| [e.g. action_taken] | Whether loan was approved | Target |
| [e.g. derived_race] | Applicant race | Protected attribute |
| [e.g. income] | Applicant income | Feature |