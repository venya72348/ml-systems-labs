# Lab 4: Data pipeline and quality

**Dataset:** UCI Adult (1994 US Census Income), loaded with `fetch_openml(name='adult', version=2, as_frame=True)`.
**Run everything with one command (from the repo root):** `python lab04/src/pipeline.py`
**Configuration:** `config.yaml` · **Full log:** `../results/pipeline.log` · **Problem definition:** `problem_definition.md` · **Data card:** `data_card.md`

| Deliverable | File |
|---|---|
| Validation report | `../results/validation_report.md`, `../results/validation_report.json` |
| Fitted preprocessor | `../results/preprocessor.joblib` |
| Processed-data hash + config + fitted values | `../results/processed_data.yaml` |
| Label audit | `../results/label_conflicts.csv`, `../results/label_sample_50.csv` |

## 1. Problem definition (summary)

The task is binary classification: predict whether a person's annual income is above $50K from 14 features (6 numeric, 8 categorical).

- **Success metrics:** F1 for the positive class `>50K` is the primary metric, and accuracy is secondary. Only 23.93% of rows are `>50K`, so a model that always predicts `<=50K` already gets about 76% accuracy. Accuracy alone therefore says little, and F1 shows whether the model actually finds high earners.
- **Constraints:** no train/test leakage, raw data is never modified, and the whole pipeline is reproducible with one command and a fixed seed (42).

The full text is in `problem_definition.md`.

## 2. Pipeline overview

```
fetch_openml ──► 1. ingest      (snapshot + SHA256, no modification)
             ──► 2. validate    (schema, ranges, categories, missing, duplicates → report, no drops)
             ──► 3. split 70/30 stratified   ◄── BEFORE any cleaning
                     │ train                                  │ test
                     ▼                                        │
                 drop exact duplicates (train only)           │ untouched
                     ▼                                        ▼
                 FIT: '?'→NaN → derived features → impute → clip P1–P99 → scale / one-hot
                     └──────────── same fitted transformer, transform() only ──────────┘
             ──► 4. LogisticRegression baseline (with / without derived features)
             ──► 5. label-quality audit
             ──► 6. SHA256 of processed data + config YAML + data card
```

## 3. Ingestion

The raw data is read once and saved unchanged to `../data/raw/adult_raw.csv`. Every later step works on copies, so the raw snapshot can always be re-hashed and compared.

- Logged shape: **48842 rows × 15 columns**, which matches the expected 48842 × 15.
- Raw SHA256: `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b`

## 4. Validation

The schema is defined in `config.yaml`:

| Check | Rule |
|---|---|
| Types | age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week = `int`; all other columns = `category` |
| Ranges | age 17–90, hours-per-week 1–99, capital-gain 0–99999, capital-loss 0–4356 |
| Categories | every value must be in the data-dictionary list for its column |
| Null policy | `?` means missing and is an allowed category |

**Result: 0 schema violation groups.** All types, ranges and categories are valid. Violations would be reported in `../results/validation_report.md`; no rows are silently dropped at this step.

**Missing values (`?`)**

| Column | Missing | % of rows |
|---|---|---|
| workclass | 2799 | 5.73% |
| occupation | 2809 | 5.75% |
| native-country | 857 | 1.75% |

The missingness is **not random**. Occupation has exactly 10 more missing values than workclass because people with `workclass = Never-worked` have no occupation by definition. In all other rows, workclass and occupation are missing together, because both come from the same unanswered employment question. Dropping these rows would remove a specific group of people (mostly not employed), so the values are **imputed** instead.

**Duplicates:** 52 exact duplicate rows (all 15 columns identical). At this step they are only reported; removal happens after the split, in train only (section 6).

## 5. Train/test split

| Split | Rows | `>50K` rate |
|---|---|---|
| Train | 34189 | 23.93% |
| Test | 14653 | 23.93% |

The split is 70/30, stratified by the label (seed 42), so both parts have the same class balance. It is done **before any cleaning**, so no statistic computed later can include test rows.

## 6. Cleaning and transformation (leakage-free)

| Step | What | Fitted on |
|---|---|---|
| 1 | `'?'` / NaN → missing | stateless |
| 2 | Derived features `capital_net`, `hours_per_age` | stateless (row-wise) |
| 3 | Drop exact duplicates: **24 rows removed** (34189 → 34165) | train only |
| 4 | Numeric imputation: median | train only |
| 5 | Categorical imputation: mode (most frequent) | train only |
| 6 | Outlier clipping to 1st / 99th percentile | train only |
| 7 | StandardScaler (numeric) | train only |
| 8 | One-hot encoding, `handle_unknown='ignore'` → 107 features | train only |

- **Duplicates:** only 24 of the 52 duplicates fell into train; the rest are in test and are kept. The test set must look like real incoming data, and removing rows there would change what we evaluate on.
- **Clipping:** extreme values (e.g. the `99999` capital-gain code, 99-hour work weeks) are limited to the train P1/P99 bounds. Test values are clipped with the **same train bounds**, never with their own percentiles.
- **Unknown categories:** if a category appears only in test, one-hot encoding gives an all-zero vector instead of an error.
- The exact fitted values (medians, modes, P1/P99 bounds per column) are stored in `../results/processed_data.yaml`.
- The fitted transformer is saved with `joblib.dump` to `../results/preprocessor.joblib`, so new data can be transformed exactly the same way.

**Leakage checklist**

| Possible leak | How it is prevented |
|---|---|
| Imputation values computed on all data | Imputers fitted on train only |
| Clip percentiles computed on all data | Clipper fitted on train only; test clipped with train bounds |
| Scaler mean/std computed on all data | Scaler fitted on train only |
| One-hot categories learned from test | Encoder fitted on train only; unknown test categories ignored |
| Deduplication across train and test | Runs inside train only, after the split |
| Derived features | Row-wise formulas with no fitted state |
| Target in features | Target column removed before the split |

## 7. Derived features

- **`capital_net = capital-gain − capital-loss`**: one signed signal for investment income instead of two sparse columns, which are 0 for most people.
- **`hours_per_age = hours-per-week / (age − 16)`**: work intensity relative to the number of possible working years. The denominator is always ≥ 1 because age ≥ 17.

**Baseline: LogisticRegression on the test set**

| Setup | Accuracy | F1 (>50K) | # features |
|---|---|---|---|
| Without derived features | 0.8540 | 0.6601 | 105 |
| With `capital_net`, `hours_per_age` | 0.8578 | 0.6696 | 107 |
| **Δ** | **+0.0038** | **+0.0095** | +2 |

**Interpretation**
- Both features improve the model slightly: F1 +0.0095, accuracy +0.0038. F1 grows more than accuracy, so the new features mainly help with the minority class `>50K`.
- The gain is small because logistic regression already sees capital-gain, capital-loss, hours and age as separate inputs. `capital_net` is almost a linear combination of two existing columns, so it adds little new information.
- `hours_per_age` is a **ratio**, which a linear model cannot build from its inputs. This non-linear structure is the most likely source of the improvement.
- Both features are computed row by row, so adding them cannot leak test information.

## 8. Label quality

Adult labels come from census answers, not from human annotators. So instead of relabelling and computing Cohen's kappa, we check label consistency and dictionary validity.

**8.1 Duplicate-conflict audit (full dataset)**

- **54 groups** of rows share all 14 feature values.
- **5 of these groups (10 rows) have conflicting income labels**: identical inputs with one row `>50K` and another `<=50K`.
- This is about 0.02% of the dataset. These rows are irreducible noise, because no model using only these features can classify both rows of a group correctly.
- They are **kept, not "fixed"**, because there is no way to know which label is correct (two real people can have the same census answers and different incomes). The full list is in `../results/label_conflicts.csv`.

**8.2 Manual re-check of 50 random rows against the data dictionary**

50 rows were drawn with seed 42 and saved to `../results/label_sample_50.csv` with a `dictionary_check` column. Each row was checked against these rules:

| # | Rule (impossible or inconsistent value) |
|---|---|
| 1 | age outside 17–90 |
| 2 | hours-per-week outside 1–99 |
| 3 | education-num does not match the dictionary value for education (e.g. Bachelors = 13) |
| 4 | relationship = Husband but sex ≠ Male |
| 5 | relationship = Wife but sex ≠ Female |
| 6 | Husband/Wife but marital-status not `Married-*` |
| 7 | capital-gain > 0 and capital-loss > 0 in the same row |
| 8 | workclass = Never-worked but an occupation is given |

**Result: 0 / 50 rows flagged.** No impossible or inconsistent values were found in the sample.

## 9. Reproducibility

The processed dataset (`../data/processed/train.csv`, `test.csv`) is written with a fixed float format and line endings, and then hashed with SHA256:

| File | SHA256 |
|---|---|
| train | `253cc30268f71e65c87aa11abb131c2ce0bea34fca999719277f66f71782b862` |
| test | `ee20c623b9330724f50ec06c25618be45c1e249ea66752de6719428be3e803b7` |
| combined | `20c9dd09b20b799a0bb43da863c49661bd562fb633058f12aeee4a0c71fe9315` |

- The hashes are stored in `../results/processed_data.yaml` together with the full config, all fitted values, the metrics and the library versions.
- **Verification:** re-run `python lab04/src/pipeline.py` and compare the hashes. The same `config.yaml` and library versions produce identical hashes, because the fixed seed (42) controls the split, the sampling and the model.
- `data/` is not committed to git, because the pipeline regenerates it.

## 10. Conclusions

1. **The raw data is structurally clean:** 0 schema violations. The real quality issues are **missing values** (not random, concentrated in workclass/occupation), **duplicates** (52) and **class imbalance** (23.93% positive).
2. **Leakage-free means "split first, fit on train only".** Every learned value (medians, modes, percentiles, scaler, categories) comes from train, and test only goes through `transform()`.
3. **Feature engineering helps a little** (F1 +0.0095), mainly through the non-linear `hours_per_age` ratio. For a linear model, new features help only when they add structure the model cannot build itself.
4. **The labels are usable:** only 5 conflicting duplicate groups and a clean 50-row dictionary check. The bigger risks are the dataset biases described in `data_card.md`: sex and race imbalance, US-centric data and the 1994 time frame.