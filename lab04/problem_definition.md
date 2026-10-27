# Problem definition

**Task.** Binary classification on the UCI Adult (1994 US Census Income) dataset: predict whether a person's annual income is above $50K.

**Inputs.** 14 features describing one adult. Six are numeric: age, fnlwgt, education-num, capital-gain, capital-loss and hours-per-week. Eight are categorical: workclass, education, marital-status, occupation, relationship, race, sex and native-country. Missing values are encoded as `?` and occur in workclass, occupation and native-country.

**Output.** `income` (OpenML column `class`), mapped to 1 for `>50K` and 0 for `<=50K`.

**Success metrics.**
- **F1 score** for the positive class `>50K`. This is the primary metric because the classes are imbalanced: about 24% of rows are positive, so a model that always predicts `<=50K` already scores about 76% accuracy.
- **Accuracy**, reported alongside F1 for comparability with the literature.

The baseline is a LogisticRegression on the processed features, evaluated on a held-out 30% stratified test set.

**Constraints.**
1. **No leakage.** The train/test split happens before any cleaning. Every learned statistic (imputation values, clipping percentiles, scaler parameters, one-hot categories) is fitted on train only and then applied unchanged to test. Duplicates are removed from train only, and the test set stays untouched.
2. **Reproducible pipeline.** One command (`python lab04/pipeline.py`) runs every step. All parameters come from `config.yaml` with a fixed seed. The raw snapshot and the processed dataset are hashed with SHA256, and the hashes are stored together with the config and library versions.
3. **Raw data is immutable.** Ingestion only reads and snapshots the data. Validation reports violations but never silently drops rows.

**Out of scope.** Model tuning and fairness mitigation. Known biases are documented in `data_card.md`.
