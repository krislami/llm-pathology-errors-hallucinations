# Manuscript-to-code analysis map

This file maps the main manuscript analyses to the outputs created by `analysis.py`.

| Manuscript analysis | Code/output |
|---|---|
| Dataset characteristics | `results/tables/table_dataset_characteristics.csv` |
| Overall and model-specific strict/partial accuracy | `results/tables/diagnostic_accuracy_summaries.csv` |
| Paired LLM accuracy comparison | `paired_binary_llm_omnibus_diagnostic.csv`; `paired_binary_llm_pairwise_diagnostic.csv` |
| Case-level coverage (≥1 correct, all four, none, only one) | `case_level_coverage_summary.csv`; `case_level_coverage_by_case.csv` |
| Accuracy by disease category, rarity, diagnostic efficiency, organ | `diagnostic_accuracy_summaries.csv` + corresponding test/post hoc CSVs |
| Overall error/hallucination prevalence and subtype frequencies | `error_hallucination_prevalence_all_summaries.csv` |
| Paired subtype comparisons across LLMs | `paired_any_*_cochranq.csv`; `paired_any_*_mcnemar.csv` |
| Error/hallucination prevalence by case characteristics | prevalence summaries + `models/gee_any_error_*` and `models/gee_any_hallucination_*` |
| Correct vs incorrect failure rates | `failures_correct_vs_incorrect.csv` + corresponding GEE outputs |
| Correct/incorrect diagnoses containing failures across LLMs | `models/gee_llm_correct_with_*`; `models/gee_llm_incorrect_with_*` |
| Error/hallucination burden | `burden_all_summaries.csv`; Friedman/Wilcoxon outputs; GEE outputs |
| Burden vs ordinal diagnostic correctness | `models/ordinal_logit_burden_vs_diagnostic_correctness.csv` |
| Clinical impact overall/by LLM/by case characteristic/by organ | `clinical_impact_summaries.csv` + test CSVs |
| Error subtype associations with clinical impact | `models/gee_errors_subtypes_*clinical_impact.csv` |
| Hallucination subtype associations with clinical impact | `models/gee_hallucinations_subtypes_*clinical_impact.csv` |
| Safe-correct / dangerous-wrong analysis | `safety_classifications.csv`; paired Cochran/McNemar outputs |
| Overall LLM performance score | `overall_score_summaries.csv` + Friedman/Wilcoxon and group comparison CSVs |
| Bayesian model: incorrect diagnosis | `models/bayesian_mixed_logit_incorrect_*` |
| Bayesian model: any error | `models/bayesian_mixed_logit_any_error_*` |
| Bayesian model: any hallucination | `models/bayesian_mixed_logit_any_hallucination_*` |
| Bayesian model: high clinical impact | `models/bayesian_mixed_logit_high_clinical_impact_*` |
| Bayesian model: low overall score | `models/bayesian_mixed_logit_low_overall_score_*` |
| All six Bayesian LLM contrasts | `models/bayesian_all_llm_pairwise_contrasts.csv` |
| Reproducibility/session information | `results/logs/session_info.txt`; `results/logs/data_checks.json` |
| Reproducible vector figures | `results/figures/*.svg` |

