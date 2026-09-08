# LLM pathology errors and hallucinations — reproducible analysis code

This repository contains the statistical analysis code for the study **“Pathology-Relevant Failure Modes and Clinical Impact of General-Purpose Large Language Models.”**

The study evaluated four general-purpose multimodal LLMs on 153 histopathology cases across 20 organs, generating 612 LLM outputs. The analysis goes beyond diagnostic accuracy to quantify pathology-relevant errors, hallucinations, cumulative burden, clinical impact, safety profiles, and factors independently associated with adverse outcomes.

## Repository contents

- `analysis.py` — complete analysis pipeline.
- `requirements.txt` — Python packages/versions used for the manuscript analysis.
- `environment.yml` — optional Conda environment.
- `data/README.md` — expected input format and data dictionary.
- `data/column_mapping_example.csv` — canonical variable names and accepted examples.
- `make_synthetic_data.py` — creates a non-sensitive synthetic dataset for a public smoke test.
- `MANUSCRIPT_ANALYSIS_MAP.md` — maps manuscript analyses to generated files.
- `CODE_AVAILABILITY_TEMPLATE.md` — manuscript-ready code-availability wording.
- `results/` — generated tables, model outputs, figures, logs, and derived variables.

## Software environment

The manuscript analyses were performed using:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- statsmodels 0.14.6
- SciPy 1.17.0

`openpyxl` is required only when the input dataset is an Excel workbook. `matplotlib` is used for reproducible SVG summary figures.

## Input data

The script expects **long format**, with one row per LLM output. In the final analysis this corresponds to 153 cases × 4 LLMs = 612 rows.

Minimum required canonical columns:

| Column | Meaning |
|---|---|
| `Case_ID` | Unique case identifier |
| `Organ` | Organ/site |
| `LLM` | `LLM1`, `LLM2`, `LLM3`, or `LLM4` |
| `Disease_Category` | `Neoplastic` or `Non-neoplastic` |
| `Rarity` | `Common`, `Sporadic`, or `Rare` |
| `Diagnostic_Efficiency` | `Fully sufficient`, `Additional support desirable`, or `Additional support required` |
| `Diagnosis_Correct` | 0=incorrect, 1=partially correct, 2=correct |
| `Omission` | Error subtype score, 0–3 |
| `Misinterpretation` | Error subtype score, 0–3 |
| `Internal_Inconsistency` | Error subtype score, 0–3 |
| `Fabricated_Feature` | Hallucination subtype score, 0–3 |
| `Unsupported_Inference` | Hallucination subtype score, 0–3 |
| `Instruction_Hallucination` | Hallucination subtype score, 0–3 |
| `Clinical_Impact` | 0–4 |
| `Overall_Score` | 1–5 |

Optional columns such as `Microscopic_Description`, `Final_Diagnosis`, and `Institution` are retained if present. If output text is available, the script also creates an exploratory response word-count analysis.

The script automatically recognizes several common alternative column spellings. See `data/column_mapping_example.csv`.

## Derived variables

The analysis script derives all manuscript outcomes from the scored subtype variables:

- **Strict diagnostic accuracy:** `Diagnosis_Correct == 2`.
- **Partial diagnostic accuracy:** `Diagnosis_Correct >= 1`.
- **Any error:** at least one error subtype score ≥1.
- **Any hallucination:** at least one hallucination subtype score ≥1.
- **Error burden:** `Omission + Misinterpretation + Internal_Inconsistency` (range 0–9).
- **Hallucination burden:** `Fabricated_Feature + Unsupported_Inference + Instruction_Hallucination` (range 0–9).
- **High error/hallucination burden:** burden ≥3.
- **High clinical impact:** score 3–4.
- **Low overall LLM performance:** score ≤2.
- **Safe correct:** strictly correct diagnosis, clinical impact ≤2, and neither error nor hallucination burden ≥3.
- **Dangerous wrong:** incorrect diagnosis, clinical impact ≥3, and error burden and/or hallucination burden ≥3.

## Statistical analyses reproduced

The pipeline reproduces the analyses described in the manuscript:

1. Counts/percentages and mean ± SD with 95% confidence intervals.
2. Wilson 95% confidence intervals for proportions.
3. Cochran's Q tests for paired binary LLM outcomes, followed by Holm-adjusted pairwise McNemar tests.
4. Fisher exact or omnibus contingency-table tests for independent categorical case characteristics.
5. Friedman tests for paired ordinal/continuous outcomes, followed by Holm-adjusted Wilcoxon signed-rank tests.
6. Mann–Whitney U or Kruskal–Wallis tests for independent case groups, with Holm-adjusted post hoc comparisons where appropriate.
7. Gaussian GEE with identity link for pooled burden/clinical-impact analyses, clustered by case.
8. Logistic GEE for pooled binary outcomes and clinically relevant correctness/failure patterns, clustered by case.
9. Per-LLM proportional-odds ordinal logistic regression of diagnostic correctness on error or hallucination burden.
10. Simultaneous subtype-impact GEE models adjusted for LLM identity.
11. Safe-correct and dangerous-wrong paired comparisons.
12. Five prespecified Bayesian mixed-effects logistic regressions for:
   - incorrect diagnosis;
   - any error;
   - any hallucination;
   - high clinical impact;
   - low overall LLM performance.
13. Bayesian models include random intercepts for case and organ. Fixed effects for the first three outcomes are LLM identity, disease category, disease rarity, and diagnostic efficiency. High clinical impact and low overall performance additionally include diagnostic correctness, total error burden, and total hallucination burden.
14. All six LLM contrasts are derived from each fitted Bayesian model with Holm-adjusted two-sided approximate Wald p values.

## Public smoke test without study data

A synthetic dataset can be generated to confirm that the environment and complete pipeline run correctly:

```bash
python make_synthetic_data.py --output data/synthetic_example.csv
python analysis.py --input data/synthetic_example.csv --output results_synthetic --strict-shape
```

The synthetic dataset contains no study observations and must not be used to verify the manuscript's numerical results.

## Run the complete analysis

Place the private analysis dataset in `data/` (or specify another location):

```bash
python analysis.py \
  --input data/LLM_Pathology_Analysis.xlsx \
  --output results \
  --strict-shape
```

For a CSV input:

```bash
python analysis.py --input data/LLM_Pathology_Analysis.csv --output results --strict-shape
```

To test the non-Bayesian part of the pipeline more quickly:

```bash
python analysis.py --input data/LLM_Pathology_Analysis.xlsx --output results --skip-bayesian
```

## Outputs

Running the script creates:

- `results/analysis_dataset_derived.csv` — canonical analysis data with all derived outcomes.
- `results/tables/` — descriptive statistics and frequentist tests.
- `results/models/` — GEE, ordinal regression, and Bayesian mixed-effects results.
- `results/figures/` — reproducible SVG summary figures.
- `results/logs/data_checks.json` — row/case/organ/LLM checks and missingness.
- `results/logs/session_info.txt` — package versions used for the run.




