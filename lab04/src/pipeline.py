"""Lab 4: Data pipeline and quality (UCI Adult).

One command:  python lab04/pipeline.py
Steps: ingest -> validate -> split -> clean/transform (train-only fit) -> derived features eval
       -> label-quality audit -> processed-data hash + config -> data card.
"""
import hashlib
import json
import logging
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # all paths below are relative to lab04/

with open("config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

P = CFG["paths"]
for d in (P["raw_dir"], P["processed_dir"], P["results_dir"]):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(os.path.join(P["results_dir"], "pipeline.log"), mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lab04")

SCHEMA = CFG["schema"]
TARGET = CFG["dataset"]["target"]
MISSING = SCHEMA["missing_token"]
FEATURES = [c for c in SCHEMA["types"] if c != TARGET]
NUM_COLS = [c for c in FEATURES if SCHEMA["types"][c] == "int"]
CAT_COLS = [c for c in FEATURES if SCHEMA["types"][c] == "category"]
SEED = CFG["seed"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_text(series):
    """Categorical column -> stripped strings, NaN shown as the missing token '?'."""
    return series.astype("object").where(series.notna(), MISSING).astype(str).str.strip()


# =============================================================== 2. Ingestion
def ingest():
    log.info("INGEST: fetch_openml(name='adult', version=2, as_frame=True)")
    bunch = fetch_openml(name=CFG["dataset"]["name"], version=CFG["dataset"]["version"], as_frame=True)
    raw = bunch.frame  # never modified below; every later step works on copies
    n_rows, n_cols = raw.shape
    log.info(f"INGEST: rows={n_rows} cols={n_cols} (expected {CFG['dataset']['expected_rows']} x {CFG['dataset']['expected_cols']})")
    if (n_rows, n_cols) != (CFG["dataset"]["expected_rows"], CFG["dataset"]["expected_cols"]):
        log.warning("INGEST: shape differs from expected")
    raw_path = os.path.join(P["raw_dir"], "adult_raw.csv")
    raw.to_csv(raw_path, index=False)  # untouched snapshot of the raw data
    raw_hash = sha256_file(raw_path)
    log.info(f"INGEST: raw snapshot {raw_path} sha256={raw_hash}")
    return raw, {"rows": n_rows, "cols": n_cols, "raw_sha256": raw_hash,
                 "openml_id": int(bunch.details.get("id", 0) or 0),
                 "licence": bunch.details.get("licence", "unknown")}


# =============================================================== 3. Validation
def validate(raw):
    df = raw.copy()
    v = {"type": [], "range": [], "category": [], "missing": {}, "duplicates": {}, "columns": {}}

    expected, actual = list(SCHEMA["types"]), list(df.columns)
    v["columns"] = {"missing_columns": [c for c in expected if c not in actual],
                    "unexpected_columns": [c for c in actual if c not in expected]}

    for col, typ in SCHEMA["types"].items():
        if col not in df:
            continue
        s = df[col]
        if typ == "int":
            num = pd.to_numeric(s, errors="coerce")
            non_numeric = int((num.isna() & s.notna()).sum())
            non_integer = int(((num % 1) != 0).sum())
            if non_numeric or non_integer:
                v["type"].append({"column": col, "expected": "int", "non_numeric": non_numeric, "non_integer": non_integer})
            v["missing"][col] = int(s.isna().sum())
        else:
            txt = as_text(s)
            v["missing"][col] = int((txt == MISSING).sum())
            allowed = set(map(str, SCHEMA["categories"].get(col, [])))
            bad = txt[~txt.isin(allowed)]
            if len(bad):
                v["category"].append({"column": col, "count": int(len(bad)),
                                      "values": {k: int(n) for k, n in bad.value_counts().head(10).items()}})

    for col, (lo, hi) in SCHEMA["ranges"].items():
        num = pd.to_numeric(df[col], errors="coerce")
        below, above = int((num < lo).sum()), int((num > hi).sum())
        v["range"].append({"column": col, "range": [lo, hi], "below": below, "above": above,
                           "min": float(num.min()), "max": float(num.max()), "ok": below == 0 and above == 0})

    txt_df = df.copy()
    for c in CAT_COLS + [TARGET]:
        txt_df[c] = as_text(df[c])
    v["duplicates"] = {"exact_duplicate_rows": int(txt_df.duplicated().sum()),
                       "duplicate_feature_rows": int(txt_df[FEATURES].duplicated().sum())}

    n_violations = len(v["type"]) + sum(not r["ok"] for r in v["range"]) + len(v["category"])
    v["summary"] = {"rows_checked": len(df), "schema_violations": n_violations,
                    "rows_with_any_missing": int((txt_df[CAT_COLS] == MISSING).any(axis=1).sum()),
                    "rows_dropped": 0}
    log.info(f"VALIDATE: {n_violations} schema violation groups; missing={ {k: m for k, m in v['missing'].items() if m} }; "
             f"exact duplicates={v['duplicates']['exact_duplicate_rows']}. No rows dropped.")

    with open(os.path.join(P["results_dir"], "validation_report.json"), "w") as f:
        json.dump(v, f, indent=2)
    return v


# =============================================================== 4-5. Transformers
class Standardize(BaseEstimator, TransformerMixin):
    """Stateless: '?' and NaN -> NaN, categoricals -> stripped str, numerics -> float."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in CAT_COLS:
            if c in X:
                t = as_text(X[c])
                X[c] = t.where(t != MISSING, np.nan)
        for c in NUM_COLS:
            if c in X:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Stateless row-wise features, so no train/test information is shared."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
        return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip to the 1st/99th percentile learned on TRAIN only."""

    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(derived=True):
    c = CFG["cleaning"]
    num = NUM_COLS + (CFG["features"]["derived"] if derived else [])
    numeric = Pipeline([("impute", SimpleImputer(strategy=c["numeric_impute"])),
                        ("clip", PercentileClipper(c["clip_lower_pct"], c["clip_upper_pct"])),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy=c["categorical_impute"])),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([
        ("standardize", Standardize()),
        ("derive", DerivedFeatures(enabled=derived)),
        ("columns", ColumnTransformer([("num", numeric, num), ("cat", categorical, CAT_COLS)])),
    ])


def labels(series):
    return (as_text(series).str.rstrip(".") == CFG["dataset"]["positive_label"]).astype(int).to_numpy()


# =============================================================== 6. Label quality
def label_audit(raw):
    txt = raw.copy()
    for c in CAT_COLS + [TARGET]:
        txt[c] = as_text(raw[c])

    # duplicate-conflict audit: identical 14 features, different income label
    g = txt.groupby(FEATURES, dropna=False, observed=True)[TARGET].nunique()
    conflict_groups = g[g > 1]
    conflict_keys = set(conflict_groups.index)
    in_conflict = txt[FEATURES].apply(tuple, axis=1).isin(conflict_keys)
    conflicts = {"duplicate_feature_groups": int((txt.groupby(FEATURES, dropna=False, observed=True).size() > 1).sum()),
                 "conflicting_groups": int(len(conflict_groups)),
                 "rows_in_conflicting_groups": int(in_conflict.sum())}
    txt[in_conflict].sort_values(FEATURES).to_csv(os.path.join(P["results_dir"], "label_conflicts.csv"), index=False)

    # 50 random rows re-checked against the data dictionary
    sample = txt.sample(n=CFG["label_audit"]["sample_size"], random_state=SEED).copy()
    edu = SCHEMA["education_num"]
    issues = []
    for _, r in sample.iterrows():
        p = []
        age, hrs = float(r["age"]), float(r["hours-per-week"])
        if not 17 <= age <= 90:
            p.append("age out of range")
        if not 1 <= hrs <= 99:
            p.append("hours out of range")
        if r["education"] in edu and int(float(r["education-num"])) != edu[r["education"]]:
            p.append("education-num does not match education")
        if r["relationship"] == "Husband" and r["sex"] != "Male":
            p.append("Husband but not Male")
        if r["relationship"] == "Wife" and r["sex"] != "Female":
            p.append("Wife but not Female")
        if r["relationship"] in ("Husband", "Wife") and not str(r["marital-status"]).startswith("Married"):
            p.append("spouse relationship but not married")
        if float(r["capital-gain"]) > 0 and float(r["capital-loss"]) > 0:
            p.append("both capital gain and loss > 0")
        if r["workclass"] == "Never-worked" and r["occupation"] != MISSING:
            p.append("Never-worked but has occupation")
        if age - 16 < 0:
            p.append("age below working age")
        issues.append("; ".join(p) if p else "ok")
    sample["dictionary_check"] = issues
    sample.to_csv(os.path.join(P["results_dir"], "label_sample_50.csv"), index=False)
    n_bad = int((sample["dictionary_check"] != "ok").sum())
    log.info(f"LABEL AUDIT: conflicting duplicate groups={conflicts['conflicting_groups']} "
             f"(rows={conflicts['rows_in_conflicting_groups']}); 50-row dictionary check: {n_bad} rows flagged")
    return conflicts, sample, n_bad


# =============================================================== Main
def main():
    raw, ingest_info = ingest()
    val = validate(raw)

    # --- split BEFORE any cleaning (stratified 70/30)
    X, y = raw[FEATURES], labels(raw[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=CFG["split"]["test_size"],
                                              stratify=y if CFG["split"]["stratify"] else None,
                                              random_state=SEED)
    log.info(f"SPLIT: train={len(X_tr)} test={len(X_te)} pos_rate train={y_tr.mean():.4f} test={y_te.mean():.4f}")

    # --- duplicates: drop exact duplicates (features + label) from TRAIN only
    n_before = len(X_tr)
    if CFG["cleaning"]["drop_train_duplicates"]:
        tr = Standardize().transform(X_tr).assign(_y=y_tr)
        keep = ~tr.duplicated()
        X_tr, y_tr = X_tr[keep.to_numpy()], y_tr[keep.to_numpy()]
    log.info(f"CLEAN: dropped {n_before - len(X_tr)} exact duplicates from train; test untouched ({len(X_te)} rows)")

    # --- fit preprocessors on TRAIN only, evaluate baseline with / without derived features
    metrics = {}
    fitted = {}
    for name, derived in [("without_derived", False), ("with_derived", True)]:
        pre = build_preprocessor(derived)
        Xt_tr = pre.fit_transform(X_tr)
        Xt_te = pre.transform(X_te)
        clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xt_tr, y_tr)
        pred = clf.predict(Xt_te)
        metrics[name] = {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
                         "f1": round(float(f1_score(y_te, pred)), 4),
                         "n_features": int(Xt_tr.shape[1])}
        fitted[name] = (pre, Xt_tr, Xt_te)
        log.info(f"BASELINE {name}: {metrics[name]}")

    pre, Xt_tr, Xt_te = fitted["with_derived"]
    joblib.dump(pre, P["preprocessor"])
    log.info(f"CLEAN: saved fitted transformer to {P['preprocessor']}")

    clip = pre.named_steps["columns"].named_transformers_["num"].named_steps["clip"]
    num_names = NUM_COLS + CFG["features"]["derived"]
    clip_bounds = {c: [round(float(a), 4), round(float(b), 4)] for c, a, b in zip(num_names, clip.lo_, clip.hi_)}
    imputer = pre.named_steps["columns"].named_transformers_["cat"].named_steps["impute"]
    cat_modes = dict(zip(CAT_COLS, map(str, imputer.statistics_)))

    # --- 7. processed dataset + SHA256
    cols = list(pre.named_steps["columns"].get_feature_names_out())
    hashes = {}
    for split, Xt, yy in [("train", Xt_tr, y_tr), ("test", Xt_te, y_te)]:
        path = os.path.join(P["processed_dir"], f"{split}.csv")
        pd.DataFrame(Xt, columns=cols).assign(income=yy).to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
        hashes[split] = sha256_file(path)
    combined = hashlib.sha256((hashes["train"] + hashes["test"]).encode()).hexdigest()
    log.info(f"HASH: train={hashes['train']} test={hashes['test']} combined={combined}")

    conflicts, sample, n_bad = label_audit(raw)

    run = {"config": CFG, "ingestion": ingest_info,
           "processed": {"train_rows": len(Xt_tr), "test_rows": len(Xt_te), "n_features": len(cols),
                         "sha256_train": hashes["train"], "sha256_test": hashes["test"], "sha256_combined": combined},
           "fitted_on_train": {"clip_bounds": clip_bounds, "categorical_modes": cat_modes},
           "metrics": metrics,
           "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                           "numpy": np.__version__, "scikit-learn": sklearn.__version__}}
    with open(os.path.join(P["results_dir"], "processed_data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run, f, sort_keys=False, allow_unicode=True)

    write_reports(raw, ingest_info, val, metrics, conflicts, sample, n_bad, run, n_before - len(X_tr))
    log.info("DONE")


# =============================================================== Reports
def md_table(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))


def write_reports(raw, ing, val, metrics, conflicts, sample, n_bad, run, n_dupes):
    R = P["results_dir"]
    # validation report
    lines = ["# Validation report", "",
             f"Rows checked: {val['summary']['rows_checked']}. Rows dropped by validation: 0 (violations are reported, never silently removed).", "",
             f"Columns: missing {val['columns']['missing_columns'] or 'none'}, unexpected {val['columns']['unexpected_columns'] or 'none'}.", "",
             "## Types", ""]
    lines += ([f"- {t['column']}: expected int, non-numeric={t['non_numeric']}, non-integer={t['non_integer']}" for t in val["type"]]
              or ["All columns match the declared types."])
    lines += ["", "## Ranges", "", "| Column | Allowed | Min | Max | Below | Above | OK |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['column']} | {r['range'][0]}–{r['range'][1]} | {r['min']:g} | {r['max']:g} | {r['below']} | {r['above']} | {'yes' if r['ok'] else 'NO'} |" for r in val["range"]]
    lines += ["", "## Categories (allowed set includes '?')", ""]
    lines += ([f"- {c['column']}: {c['count']} values outside the dictionary: {c['values']}" for c in val["category"]]
              or ["All categorical values are in the allowed sets."])
    lines += ["", "## Missing values ('?' or NaN)", "", "| Column | Missing | % |", "|---|---|---|"]
    lines += [f"| {k} | {m} | {100*m/len(raw):.2f}% |" for k, m in val["missing"].items() if m]
    lines += ["", f"Rows with at least one missing categorical value: {val['summary']['rows_with_any_missing']}.", "",
              "## Duplicates", "",
              f"- Exact duplicate rows (all 15 columns): {val['duplicates']['exact_duplicate_rows']}",
              f"- Duplicate feature rows (14 features): {val['duplicates']['duplicate_feature_rows']}",
              "", f"Total schema violation groups: **{val['summary']['schema_violations']}**."]
    open(os.path.join(R, "validation_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # main report
    m0, m1 = metrics["without_derived"], metrics["with_derived"]
    flagged = sample[sample["dictionary_check"] != "ok"][["age", "education", "education-num", "relationship", "sex", "dictionary_check"]]
    rep = f"""# Lab 4: Data pipeline and quality

Run with one command: `python lab04/pipeline.py`. Full log: `results/pipeline.log`. Problem definition: `problem_definition.md`. Data card: `data_card.md`.

## 1. Ingestion
Raw data read with `fetch_openml(name='adult', version=2, as_frame=True)` and saved unchanged to `data/raw/adult_raw.csv`.
Logged shape: **{ing['rows']} rows × {ing['cols']} columns**. Raw SHA256: `{ing['raw_sha256']}`.

## 2. Validation
See `results/validation_report.md` (and `.json`). Schema violation groups: **{val['summary']['schema_violations']}**. Exact duplicates: {val['duplicates']['exact_duplicate_rows']}. No rows are dropped at this step.

## 3. Cleaning and transformation (leakage-free)
1. Stratified 70/30 split **before** any cleaning (seed {SEED}).
2. {n_dupes} exact duplicates dropped from **train only**; test kept untouched.
3. `'?'` → missing; median imputation (numeric) and mode imputation (categorical) **fitted on train only**.
4. Numeric clipping to train 1st/99th percentiles (bounds in `results/processed_data.yaml`).
5. Standard scaling + one-hot encoding (unknown test categories ignored).
6. Fitted transformer saved with `joblib.dump` to `results/preprocessor.joblib`.

All statistics (medians, modes, percentiles, scaler means, category lists) come from train data; test only passes through `transform`.

## 4. Derived features (LogisticRegression baseline, test set)

| Setup | Accuracy | F1 (>50K) | # features |
|---|---|---|---|
| Without derived | {m0['accuracy']:.4f} | {m0['f1']:.4f} | {m0['n_features']} |
| With `capital_net`, `hours_per_age` | {m1['accuracy']:.4f} | {m1['f1']:.4f} | {m1['n_features']} |

`capital_net = capital-gain − capital-loss` combines two sparse columns into one signed signal. `hours_per_age = hours-per-week / (age − 16)` measures work intensity relative to working years. Both are row-wise, so they cannot leak test information. The change in metrics is {'positive' if m1['f1'] > m0['f1'] else 'negligible or negative'} (ΔF1 = {m1['f1'] - m0['f1']:+.4f}): a linear model already sees capital-gain/loss, hours and age separately, so the derived features mostly re-express existing information.

## 5. Label quality
**Duplicate-conflict audit:** {conflicts['duplicate_feature_groups']} groups of rows share all 14 features; **{conflicts['conflicting_groups']}** of them have conflicting income labels ({conflicts['rows_in_conflicting_groups']} rows, listed in `results/label_conflicts.csv`). These are irreducible noise for any model using only these features.

**50-row dictionary check:** 50 random rows (seed {SEED}) saved to `results/label_sample_50.csv` with a `dictionary_check` column. Checked: age/hours ranges, education ↔ education-num mapping, Husband/Wife ↔ sex, spouse relationship ↔ married status, gain and loss both > 0, Never-worked with an occupation. Rows flagged: **{n_bad}/50**.

{md_table(flagged) if len(flagged) else 'No impossible values found in the sample.'}

## 6. Reproducibility
Processed dataset (`data/processed/train.csv`, `test.csv`) SHA256:
- train: `{run['processed']['sha256_train']}`
- test: `{run['processed']['sha256_test']}`
- combined: `{run['processed']['sha256_combined']}`

Stored together with the full config, fitted bounds, metrics and library versions in `results/processed_data.yaml`. Re-running `python lab04/pipeline.py` with the same `config.yaml` and library versions reproduces the same hashes.
"""
    open("report.md", "w", encoding="utf-8").write(rep)

    # data card
    def dist(col):
        s = as_text(raw[col]).value_counts(normalize=True).head(5)
        return ", ".join(f"{k} {100*v:.1f}%" for k, v in s.items())

    pos = labels(raw[TARGET]).mean()
    by_sex = pd.Series(labels(raw[TARGET])).groupby(as_text(raw["sex"]).to_numpy()).mean()
    card = f"""# Data card: UCI Adult (Census Income)

## Source
- UCI Machine Learning Repository, "Adult" (Becker & Kohavi, 1996), extracted by Barry Becker from the **1994 US Census** database.
- Accessed via OpenML: `fetch_openml(name='adult', version=2)` (OpenML id {ing['openml_id']}).
- Raw snapshot SHA256: `{ing['raw_sha256']}`.

## License
{ing['licence']} as listed on OpenML; UCI distributes the dataset under **CC BY 4.0**. Attribution: Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Size and structure
- {ing['rows']} rows × {ing['cols']} columns (14 features: 6 numeric, 8 categorical; target `income`/`class`).
- Processed: {run['processed']['train_rows']} train rows (after dropping train duplicates), {run['processed']['test_rows']} test rows, {run['processed']['n_features']} model features.
- Missing values (`?`): workclass {val['missing'].get('workclass', 0)}, occupation {val['missing'].get('occupation', 0)}, native-country {val['missing'].get('native-country', 0)}.

## Intended use
Teaching and benchmarking binary classification (income > $50K/year). **Not** for real decisions about individuals (credit, hiring, insurance).

## Known biases and limitations
- **Label imbalance:** only {100*pos:.1f}% of rows are `>50K`, so accuracy alone is misleading; F1 is reported.
- **Sex:** {dist('sex')}. Positive rate by sex: {', '.join(f'{k} {100*v:.1f}%' for k, v in by_sex.items())}. A model will learn this gap.
- **Race:** {dist('race')}. Minority groups have few samples, so per-group error is unreliable.
- **Geography:** native-country is dominated by the US ({100*(as_text(raw['native-country']) == 'United-States').mean():.1f}% United-States); results do not generalise to other countries.
- **Time:** 1994 data; the $50K threshold is not inflation-adjusted and occupations/wages have changed.
- `fnlwgt` is a census sampling weight, not a personal attribute, and is easy to misuse as a feature.
- Missing values are not random (e.g. occupation is missing when workclass is missing / Never-worked).
- {conflicts['conflicting_groups']} duplicate feature groups have conflicting labels (label noise).

## Privacy concerns
- Records are anonymised census extracts with no names or IDs, but they contain **sensitive attributes** (race, sex, marital status, native country, income).
- Rare combinations (e.g. small native-country + occupation + age) could make re-identification easier; do not join with other datasets to identify people.
- Sensitive attributes should be used only for fairness auditing, not to make decisions.

## Pipeline
Leakage-free preprocessing (split first, train-only fitting), saved transformer `results/preprocessor.joblib`, config `config.yaml`, processed hashes in `results/processed_data.yaml`. Reproduce with `python lab04/pipeline.py`.
"""
    open("data_card.md", "w", encoding="utf-8").write(card)


if __name__ == "__main__":
    main()
