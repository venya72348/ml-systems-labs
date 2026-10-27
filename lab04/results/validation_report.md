# Validation report

Rows checked: 48842. Rows dropped by validation: 0 (violations are reported, never silently removed).

Columns: missing none, unexpected none.

## Types

All columns match the declared types.

## Ranges

| Column | Allowed | Min | Max | Below | Above | OK |
|---|---|---|---|---|---|---|
| age | 17–90 | 17 | 90 | 0 | 0 | yes |
| hours-per-week | 1–99 | 1 | 99 | 0 | 0 | yes |
| capital-gain | 0–99999 | 0 | 99999 | 0 | 0 | yes |
| capital-loss | 0–4356 | 0 | 4356 | 0 | 0 | yes |

## Categories (allowed set includes '?')

All categorical values are in the allowed sets.

## Missing values ('?' or NaN)

| Column | Missing | % |
|---|---|---|
| workclass | 2799 | 5.73% |
| occupation | 2809 | 5.75% |
| native-country | 857 | 1.75% |

Rows with at least one missing categorical value: 3620.

## Duplicates

- Exact duplicate rows (all 15 columns): 52
- Duplicate feature rows (14 features): 57

Total schema violation groups: **0**.
