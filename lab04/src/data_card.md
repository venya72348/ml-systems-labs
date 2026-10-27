# Data card: UCI Adult (Census Income)

## Source
- UCI Machine Learning Repository, "Adult" (Becker & Kohavi, 1996), extracted by Barry Becker from the **1994 US Census** database.
- Accessed via OpenML: `fetch_openml(name='adult', version=2)` (OpenML id 1590).
- Raw snapshot SHA256: `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b`.

## License
Public as listed on OpenML; UCI distributes the dataset under **CC BY 4.0**. Attribution: Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Size and structure
- 48842 rows × 15 columns (14 features: 6 numeric, 8 categorical; target `income`/`class`).
- Processed: 34165 train rows (after dropping train duplicates), 14653 test rows, 107 model features.
- Missing values (`?`): workclass 2799, occupation 2809, native-country 857.

## Intended use
Teaching and benchmarking binary classification (income > $50K/year). **Not** for real decisions about individuals (credit, hiring, insurance).

## Known biases and limitations
- **Label imbalance:** only 23.9% of rows are `>50K`, so accuracy alone is misleading; F1 is reported.
- **Sex:** Male 66.8%, Female 33.2%. Positive rate by sex: Female 10.9%, Male 30.4%. A model will learn this gap.
- **Race:** White 85.5%, Black 9.6%, Asian-Pac-Islander 3.1%, Amer-Indian-Eskimo 1.0%, Other 0.8%. Minority groups have few samples, so per-group error is unreliable.
- **Geography:** native-country is dominated by the US (89.7% United-States); results do not generalise to other countries.
- **Time:** 1994 data; the $50K threshold is not inflation-adjusted and occupations/wages have changed.
- `fnlwgt` is a census sampling weight, not a personal attribute, and is easy to misuse as a feature.
- Missing values are not random (e.g. occupation is missing when workclass is missing / Never-worked).
- 5 duplicate feature groups have conflicting labels (label noise).

## Privacy concerns
- Records are anonymised census extracts with no names or IDs, but they contain **sensitive attributes** (race, sex, marital status, native country, income).
- Rare combinations (e.g. small native-country + occupation + age) could make re-identification easier; do not join with other datasets to identify people.
- Sensitive attributes should be used only for fairness auditing, not to make decisions.

## Pipeline
Leakage-free preprocessing (split first, train-only fitting), saved transformer `results/preprocessor.joblib`, config `config.yaml`, processed hashes in `results/processed_data.yaml`. Reproduce with `python lab04/pipeline.py`.
