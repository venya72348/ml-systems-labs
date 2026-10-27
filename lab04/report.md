# Lab 4: Data pipeline and quality

Run with one command: `python lab04/pipeline.py`. Full log: `results/pipeline.log`. Problem definition: `problem_definition.md`. Data card: `data_card.md`.

## 1. Ingestion
Raw data read with `fetch_openml(name='adult', version=2, as_frame=True)` and saved unchanged to `data/raw/adult_raw.csv`.
Logged shape: **48842 rows × 15 columns**. Raw SHA256: `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b`.

## 2. Validation
See `results/validation_report.md` (and `.json`). Schema violation groups: **0**. Exact duplicates: 52. No rows are dropped at this step.

## 3. Cleaning and transformation (leakage-free)
1. Stratified 70/30 split **before** any cleaning (seed 42).
2. 24 exact duplicates dropped from **train only**; test kept untouched.
3. `'?'` → missing; median imputation (numeric) and mode imputation (categorical) **fitted on train only**.
4. Numeric clipping to train 1st/99th percentiles (bounds in `results/processed_data.yaml`).
5. Standard scaling + one-hot encoding (unknown test categories ignored).
6. Fitted transformer saved with `joblib.dump` to `results/preprocessor.joblib`.

All statistics (medians, modes, percentiles, scaler means, category lists) come from train data; test only passes through `transform`.

## 4. Derived features (LogisticRegression baseline, test set)

| Setup | Accuracy | F1 (>50K) | # features |
|---|---|---|---|
| Without derived | 0.8540 | 0.6601 | 105 |
| With `capital_net`, `hours_per_age` | 0.8578 | 0.6696 | 107 |

`capital_net = capital-gain − capital-loss` combines two sparse columns into one signed signal. `hours_per_age = hours-per-week / (age − 16)` measures work intensity relative to working years. Both are row-wise, so they cannot leak test information. The change in metrics is positive (ΔF1 = +0.0095): a linear model already sees capital-gain/loss, hours and age separately, so the derived features mostly re-express existing information.

## 5. Label quality
**Duplicate-conflict audit:** 54 groups of rows share all 14 features; **5** of them have conflicting income labels (10 rows, listed in `results/label_conflicts.csv`). These are irreducible noise for any model using only these features.

**50-row dictionary check:** 50 random rows (seed 42) saved to `results/label_sample_50.csv` with a `dictionary_check` column. Checked: age/hours ranges, education ↔ education-num mapping, Husband/Wife ↔ sex, spouse relationship ↔ married status, gain and loss both > 0, Never-worked with an occupation. Rows flagged: **0/50**.

No impossible values found in the sample.

## 6. Reproducibility
Processed dataset (`data/processed/train.csv`, `test.csv`) SHA256:
- train: `253cc30268f71e65c87aa11abb131c2ce0bea34fca999719277f66f71782b862`
- test: `ee20c623b9330724f50ec06c25618be45c1e249ea66752de6719428be3e803b7`
- combined: `20c9dd09b20b799a0bb43da863c49661bd562fb633058f12aeee4a0c71fe9315`

Stored together with the full config, fitted bounds, metrics and library versions in `results/processed_data.yaml`. Re-running `python lab04/pipeline.py` with the same `config.yaml` and library versions reproduces the same hashes.
