#!/usr/bin/env python3
"""
Reproducible statistical analysis for:
"Pathology-Relevant Failure Modes and Clinical Impact of General-Purpose Large Language Models"

This script implements the analyses described in the manuscript statistical-analysis section.
It expects one row per LLM output (153 cases x 4 LLMs = 612 rows in the final study dataset).

Primary analyses implemented
----------------------------
1. Dataset characteristics.
2. Strict and partial diagnostic accuracy, overall/by LLM/by case characteristic/by organ.
3. Case-level diagnostic coverage across the four LLMs.
4. Prevalence of errors, hallucinations, and their six subtypes.
5. Paired binary comparisons across LLMs (Cochran's Q + Holm-adjusted McNemar tests).
6. Error/hallucination burden (0-9), including Friedman/Wilcoxon and pooled GEE analyses.
7. Proportional-odds ordinal logistic regression relating burden to diagnostic correctness.
8. Clinical impact analyses, including subtype-impact GEE models.
9. Clinically relevant output patterns and safe-correct/dangerous-wrong classifications.
10. Overall LLM performance score analyses.
11. Five prespecified Bayesian mixed-effects logistic regressions with case and organ random intercepts.
12. Approximate joint Wald and all six pairwise LLM contrasts with Holm-adjusted p values.
13. Main/supplementary summary figures and machine-readable CSV outputs.
14. Optional exploratory response-length summaries if text columns are available.

Definitions used in the manuscript
----------------------------------
- Diagnosis_Correct: 0=incorrect, 1=partially correct, 2=correct.
- Any error: >=1 in omission, misinterpretation, or internal inconsistency.
- Any hallucination: >=1 in fabricated feature, unsupported inference, or instruction hallucination.
- Error burden: sum of the three error subtype scores (0-9).
- Hallucination burden: sum of the three hallucination subtype scores (0-9).
- High burden: burden >=3.
- High clinical impact: Clinical_Impact >=3.
- Low overall LLM performance: Overall_Score <=2.
- Safe correct: strictly correct, Clinical_Impact <=2, and neither error nor hallucination burden is high.
- Dangerous wrong: incorrect, Clinical_Impact >=3, and high error burden and/or high hallucination burden.

Usage
-----
python analysis.py --input data/LLM_Pathology_Analysis.xlsx --output results

The input may also be CSV. Column aliases are normalized automatically; see README.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LLM_ORDER = ["LLM1", "LLM2", "LLM3", "LLM4"]
RARITY_ORDER = ["Common", "Sporadic", "Rare"]
EFFICIENCY_ORDER = ["Fully sufficient", "Additional support desirable", "Additional support required"]
CATEGORY_ORDER = ["Neoplastic", "Non-neoplastic"]
DIAGNOSIS_ORDER = [0, 1, 2]

ERROR_SUBTYPES = ["Omission", "Misinterpretation", "Internal_Inconsistency"]
HALLUCINATION_SUBTYPES = ["Fabricated_Feature", "Unsupported_Inference", "Instruction_Hallucination"]
ALL_SUBTYPES = ERROR_SUBTYPES + HALLUCINATION_SUBTYPES

REQUIRED_COLUMNS = [
    "Case_ID", "Organ", "LLM", "Disease_Category", "Rarity", "Diagnostic_Efficiency",
    "Diagnosis_Correct", *ALL_SUBTYPES, "Clinical_Impact", "Overall_Score",
]

# Flexible aliases for common versions used during data assembly.
COLUMN_ALIASES = {
    "Case_ID": ["case_id", "caseid", "case", "id"],
    "Organ": ["organ", "organ_system", "organ type", "organ_type"],
    "LLM": ["llm", "model", "llm_id", "model_id"],
    "Disease_Category": ["disease_category", "category", "disease class", "disease_class"],
    "Rarity": ["rarity", "disease_rarity"],
    "Diagnostic_Efficiency": ["diagnostic_efficiency", "image_diagnostic_efficiency", "efficiency", "diagnostic efficiency"],
    "Diagnosis_Correct": ["diagnosis_correct", "diagnostic_correctness", "correctness", "diagnosis correctness", "diagnosis_score"],
    "Omission": ["omission", "e1", "error_omission"],
    "Misinterpretation": ["misinterpretation", "e2", "error_misinterpretation"],
    "Internal_Inconsistency": ["internal_inconsistency", "inconsistency", "e3", "internal inconsistency"],
    "Fabricated_Feature": ["fabricated_feature", "fabricated_features", "h1", "fabricated feature"],
    "Unsupported_Inference": ["unsupported_inference", "unsupported_inferences", "h2", "unsupported inference"],
    "Instruction_Hallucination": ["instruction_hallucination", "instruction_hallucinations", "h3", "instruction hallucination"],
    "Clinical_Impact": ["clinical_impact", "clinical impact", "impact", "clinical_impact_score"],
    "Overall_Score": ["overall_score", "overall llm score", "overall_llm_score", "overall performance", "overall_performance"],
    "Microscopic_Description": ["microscopic_description", "microscopic description", "description", "llm_description"],
    "Final_Diagnosis": ["final_diagnosis", "final diagnosis", "llm_final_diagnosis", "llm_diagnosis"],
    "Institution": ["institution", "site", "center", "centre"],
}

@dataclass
class AnalysisPaths:
    root: Path
    tables: Path
    models: Path
    figures: Path
    logs: Path

    @classmethod
    def create(cls, root: str | Path) -> "AnalysisPaths":
        root = Path(root)
        paths = cls(
            root=root,
            tables=root / "tables",
            models=root / "models",
            figures=root / "figures",
            logs=root / "logs",
        )
        for p in [paths.root, paths.tables, paths.models, paths.figures, paths.logs]:
            p.mkdir(parents=True, exist_ok=True)
        return paths

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def clean_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to the canonical column names used by this script."""
    current = {clean_name(c): c for c in df.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = [canonical] + aliases
        for candidate in candidates:
            key = clean_name(candidate)
            if key in current:
                rename[current[key]] = canonical
                break
    return df.rename(columns=rename)


def load_input(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def standardize_llm(x) -> str:
    s = clean_name(x)
    mapping = {
        "llm1": "LLM1", "1": "LLM1", "gpt_5_3": "LLM1", "gpt5_3": "LLM1", "chatgpt_5_3": "LLM1",
        "llm2": "LLM2", "2": "LLM2", "gemini_3": "LLM2", "gemini3": "LLM2",
        "llm3": "LLM3", "3": "LLM3", "grok_4_20": "LLM3", "grok_4_3": "LLM3", "grok4_20": "LLM3", "grok4_3": "LLM3",
        "llm4": "LLM4", "4": "LLM4", "opus_4_6": "LLM4", "claude_opus_4_6": "LLM4", "claude_4_6": "LLM4",
    }
    return mapping.get(s, str(x).strip())


def standardize_category(x) -> str:
    s = clean_name(x)
    if s in {"neoplastic", "neoplasm", "tumor", "tumour"}:
        return "Neoplastic"
    if s in {"non_neoplastic", "nonneoplastic", "non_neoplasm", "non_tumor", "non_tumour"}:
        return "Non-neoplastic"
    return str(x).strip()


def standardize_rarity(x) -> str:
    s = clean_name(x)
    if s in {"common", "1"}: return "Common"
    if s in {"sporadic", "2", "uncommon"}: return "Sporadic"
    if s in {"rare", "3"}: return "Rare"
    return str(x).strip()


def standardize_efficiency(x) -> str:
    s = clean_name(x)
    if s in {"fully_sufficient", "sufficient", "fully_sufficient_for_diagnosis", "1"}:
        return "Fully sufficient"
    if s in {"additional_support_desirable", "support_desirable", "desirable", "2"}:
        return "Additional support desirable"
    if s in {"additional_support_required", "support_required", "required", "3"}:
        return "Additional support required"
    return str(x).strip()


def standardize_correctness(x) -> int | float:
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    s = clean_name(x)
    if s in {"0", "incorrect", "wrong"}: return 0
    if s in {"1", "partial", "partially_correct", "partially"}: return 1
    if s in {"2", "correct", "strictly_correct"}: return 2
    raise ValueError(f"Unrecognized Diagnosis_Correct value: {x!r}")


def validate_and_derive(df: pd.DataFrame, strict_shape: bool = False) -> pd.DataFrame:
    df = normalize_columns(df).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nSee README.md for the expected long-format data dictionary and aliases."
        )

    df["Case_ID"] = df["Case_ID"].astype(str)
    df["Organ"] = df["Organ"].astype(str).str.strip()
    df["LLM"] = df["LLM"].map(standardize_llm)
    df["Disease_Category"] = df["Disease_Category"].map(standardize_category)
    df["Rarity"] = df["Rarity"].map(standardize_rarity)
    df["Diagnostic_Efficiency"] = df["Diagnostic_Efficiency"].map(standardize_efficiency)
    df["Diagnosis_Correct"] = df["Diagnosis_Correct"].map(standardize_correctness)

    numeric_cols = ALL_SUBTYPES + ["Clinical_Impact", "Overall_Score"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Range validation according to the study scoring framework.
    for c in ALL_SUBTYPES:
        bad = ~df[c].between(0, 3)
        if bad.any(): raise ValueError(f"{c} contains values outside 0-3.")
    if (~df["Diagnosis_Correct"].isin([0, 1, 2])).any():
        raise ValueError("Diagnosis_Correct must be 0, 1, or 2.")
    if (~df["Clinical_Impact"].between(0, 4)).any():
        raise ValueError("Clinical_Impact must be 0-4.")
    if (~df["Overall_Score"].between(1, 5)).any():
        raise ValueError("Overall_Score must be 1-5.")

    if strict_shape:
        if len(df) != 612:
            raise ValueError(f"Expected 612 LLM outputs in final dataset, found {len(df)}.")
        if df["Case_ID"].nunique() != 153:
            raise ValueError(f"Expected 153 cases, found {df['Case_ID'].nunique()}.")
        per_case = df.groupby("Case_ID")["LLM"].nunique()
        if not (per_case == 4).all():
            raise ValueError("Every case must contain exactly four unique LLM outputs.")

    # Canonical categorical ordering.
    df["LLM"] = pd.Categorical(df["LLM"], categories=LLM_ORDER, ordered=True)
    df["Disease_Category"] = pd.Categorical(df["Disease_Category"], categories=CATEGORY_ORDER, ordered=True)
    df["Rarity"] = pd.Categorical(df["Rarity"], categories=RARITY_ORDER, ordered=True)
    df["Diagnostic_Efficiency"] = pd.Categorical(df["Diagnostic_Efficiency"], categories=EFFICIENCY_ORDER, ordered=True)

    # Derived variables used throughout the manuscript.
    df["Strict_Correct"] = (df["Diagnosis_Correct"] == 2).astype(int)
    df["Partial_or_Correct"] = (df["Diagnosis_Correct"] >= 1).astype(int)
    df["Incorrect"] = (df["Diagnosis_Correct"] == 0).astype(int)

    for c in ALL_SUBTYPES:
        df[f"Any_{c}"] = (df[c] >= 1).astype(int)

    df["Any_Error"] = (df[ERROR_SUBTYPES].max(axis=1) >= 1).astype(int)
    df["Any_Hallucination"] = (df[HALLUCINATION_SUBTYPES].max(axis=1) >= 1).astype(int)
    df["Any_Failure"] = ((df["Any_Error"] == 1) | (df["Any_Hallucination"] == 1)).astype(int)
    df["Error_Burden"] = df[ERROR_SUBTYPES].sum(axis=1)
    df["Hallucination_Burden"] = df[HALLUCINATION_SUBTYPES].sum(axis=1)
    df["Total_Failure_Burden"] = df["Error_Burden"] + df["Hallucination_Burden"]
    df["High_Error_Burden"] = (df["Error_Burden"] >= 3).astype(int)
    df["High_Hallucination_Burden"] = (df["Hallucination_Burden"] >= 3).astype(int)
    df["High_Clinical_Impact"] = (df["Clinical_Impact"] >= 3).astype(int)
    df["Low_Overall_Score"] = (df["Overall_Score"] <= 2).astype(int)

    df["Correct_with_Any_Error"] = ((df["Diagnosis_Correct"] == 2) & (df["Any_Error"] == 1)).astype(int)
    df["Correct_with_Any_Hallucination"] = ((df["Diagnosis_Correct"] == 2) & (df["Any_Hallucination"] == 1)).astype(int)
    df["Correct_with_Any_Failure"] = ((df["Diagnosis_Correct"] == 2) & (df["Any_Failure"] == 1)).astype(int)
    df["Incorrect_with_Any_Error"] = ((df["Diagnosis_Correct"] == 0) & (df["Any_Error"] == 1)).astype(int)
    df["Incorrect_with_Any_Hallucination"] = ((df["Diagnosis_Correct"] == 0) & (df["Any_Hallucination"] == 1)).astype(int)
    df["Incorrect_with_Any_Failure"] = ((df["Diagnosis_Correct"] == 0) & (df["Any_Failure"] == 1)).astype(int)

    df["Safe_Correct"] = (
        (df["Diagnosis_Correct"] == 2)
        & (df["Clinical_Impact"] <= 2)
        & (df["High_Error_Burden"] == 0)
        & (df["High_Hallucination_Burden"] == 0)
    ).astype(int)
    df["Dangerous_Wrong"] = (
        (df["Diagnosis_Correct"] == 0)
        & (df["Clinical_Impact"] >= 3)
        & ((df["High_Error_Burden"] == 1) | (df["High_Hallucination_Burden"] == 1))
    ).astype(int)

    # Optional response-length variables if raw text is present.
    text_cols = [c for c in ["Microscopic_Description", "Final_Diagnosis"] if c in df.columns]
    if text_cols:
        for c in text_cols:
            df[f"{c}_Word_Count"] = df[c].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
        df["Response_Word_Count"] = sum(df[f"{c}_Word_Count"] for c in text_cols)

    return df


def mean_ci(x: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float, float, int]:
    s = pd.Series(x).dropna().astype(float)
    n = len(s)
    mean = s.mean() if n else np.nan
    sd = s.std(ddof=1) if n > 1 else np.nan
    if n > 1:
        se = sd / math.sqrt(n)
        z = stats.norm.ppf(0.5 + confidence / 2)
        lo, hi = mean - z * se, mean + z * se
    else:
        lo = hi = np.nan
    return mean, sd, lo, hi, n


def proportion_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    z = stats.norm.ppf(0.5 + confidence / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * math.sqrt((p*(1-p)/n) + z**2/(4*n**2)) / denom
    return p, center-half, center+half


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def save_df(df: pd.DataFrame, path: Path, index: bool = False):
    df.to_csv(path, index=index)


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def complete_cases(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df.dropna(subset=list(cols)).copy()

# -----------------------------------------------------------------------------
# Descriptive tables
# -----------------------------------------------------------------------------

def dataset_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    case = df.sort_values(["Case_ID", "LLM"]).drop_duplicates("Case_ID")
    rows = []
    rows.append({"Characteristic": "Cases", "Category": "Total", "N": case["Case_ID"].nunique(), "Percent": 100.0})
    rows.append({"Characteristic": "LLM outputs", "Category": "Total", "N": len(df), "Percent": 100.0})
    rows.append({"Characteristic": "Organs", "Category": "Total unique", "N": case["Organ"].nunique(), "Percent": np.nan})
    for var in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
        counts = case[var].value_counts(dropna=False, sort=False)
        for cat, n in counts.items():
            rows.append({"Characteristic": var, "Category": str(cat), "N": int(n), "Percent": 100*n/len(case)})
    if "Institution" in case.columns:
        for cat, n in case["Institution"].value_counts(dropna=False).items():
            rows.append({"Characteristic": "Institution", "Category": str(cat), "N": int(n), "Percent": 100*n/len(case)})
    return pd.DataFrame(rows)


def binary_summary(df: pd.DataFrame, outcome: str, by: Sequence[str] | None = None) -> pd.DataFrame:
    by = list(by or [])
    grouper = by[0] if len(by) == 1 else by
    rows = []
    grouped = [((), df)] if not by else df.groupby(grouper, observed=True, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple): keys = (keys,)
        n = len(g); k = int(g[outcome].sum())
        p, lo, hi = proportion_ci(k, n)
        row = dict(zip(by, keys))
        row.update({"Outcome": outcome, "N": n, "Events": k, "Proportion": p, "Percent": 100*p, "CI95_Lower": lo, "CI95_Upper": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def continuous_summary(df: pd.DataFrame, outcome: str, by: Sequence[str] | None = None) -> pd.DataFrame:
    by = list(by or [])
    grouper = by[0] if len(by) == 1 else by
    rows = []
    grouped = [((), df)] if not by else df.groupby(grouper, observed=True, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple): keys = (keys,)
        mean, sd, lo, hi, n = mean_ci(g[outcome])
        row = dict(zip(by, keys))
        row.update({"Outcome": outcome, "N": n, "Mean": mean, "SD": sd, "CI95_Lower": lo, "CI95_Upper": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def case_level_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = df.pivot_table(index="Case_ID", columns="LLM", values="Strict_Correct", aggfunc="first")
    p = p.reindex(columns=LLM_ORDER)
    p["N_Correct_LLMs"] = p.sum(axis=1)
    p["At_least_one_correct"] = (p["N_Correct_LLMs"] >= 1).astype(int)
    p["All_four_correct"] = (p["N_Correct_LLMs"] == 4).astype(int)
    p["None_correct"] = (p["N_Correct_LLMs"] == 0).astype(int)
    p["Exactly_one_correct"] = (p["N_Correct_LLMs"] == 1).astype(int)

    summary = []
    n = len(p)
    for label, col in [
        ("At least one LLM correct", "At_least_one_correct"),
        ("All four LLMs correct", "All_four_correct"),
        ("No LLM correct", "None_correct"),
        ("Exactly one LLM correct", "Exactly_one_correct"),
    ]:
        k = int(p[col].sum())
        summary.append({"Metric": label, "N": k, "Total": n, "Percent": 100*k/n})
    if p["Exactly_one_correct"].sum():
        one = p[p["Exactly_one_correct"] == 1][LLM_ORDER]
        for llm in LLM_ORDER:
            summary.append({"Metric": f"Only {llm} correct", "N": int(one[llm].sum()), "Total": n, "Percent": 100*one[llm].sum()/n})
    return pd.DataFrame(summary), p.reset_index()

# -----------------------------------------------------------------------------
# Frequentist tests
# -----------------------------------------------------------------------------

def paired_binary_llm_tests(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = df.pivot_table(index="Case_ID", columns="LLM", values=outcome, aggfunc="first").reindex(columns=LLM_ORDER).dropna()
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = cochrans_q(wide.to_numpy(dtype=int))
    omnibus = pd.DataFrame([{"Outcome": outcome, "Test": "Cochran Q", "Statistic": float(q.statistic), "df": len(LLM_ORDER)-1, "p": float(q.pvalue)}])
    rows = []
    for a, b in itertools.combinations(LLM_ORDER, 2):
        tab = pd.crosstab(wide[a], wide[b]).reindex(index=[0,1], columns=[0,1], fill_value=0)
        res = mcnemar(tab.to_numpy(), exact=True)
        rows.append({"Outcome": outcome, "LLM_A": a, "LLM_B": b, "Statistic": float(res.statistic), "p_raw": float(res.pvalue)})
    pair = pd.DataFrame(rows)
    pair["p_holm"] = holm_adjust(pair["p_raw"])
    pair["Significance"] = pair["p_holm"].map(stars)
    return omnibus, pair


def independent_binary_test(df: pd.DataFrame, outcome: str, group: str, per_llm: bool = False) -> pd.DataFrame:
    rows = []
    subsets = [("Overall", df)] if not per_llm else [(str(k), g) for k, g in df.groupby("LLM", observed=True)]
    for label, g in subsets:
        tab = pd.crosstab(g[group], g[outcome])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            rows.append({"Scope": label, "Outcome": outcome, "Group": group, "Test": "Not estimable", "Statistic": np.nan, "p": np.nan})
            continue
        if tab.shape == (2, 2):
            odds, p = stats.fisher_exact(tab.to_numpy())
            rows.append({"Scope": label, "Outcome": outcome, "Group": group, "Test": "Fisher exact", "Statistic": odds, "p": p})
        else:
            chi2, p, dof, _ = stats.chi2_contingency(tab.to_numpy())
            rows.append({"Scope": label, "Outcome": outcome, "Group": group, "Test": "Chi-square", "Statistic": chi2, "df": dof, "p": p})
    out = pd.DataFrame(rows)
    if per_llm and not out.empty:
        out["p_holm"] = holm_adjust(out["p"])
    return out


def paired_continuous_llm_tests(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = df.pivot_table(index="Case_ID", columns="LLM", values=outcome, aggfunc="first").reindex(columns=LLM_ORDER).dropna()
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    fr = stats.friedmanchisquare(*(wide[c].to_numpy() for c in LLM_ORDER))
    omnibus = pd.DataFrame([{"Outcome": outcome, "Test": "Friedman", "Statistic": fr.statistic, "df": 3, "p": fr.pvalue}])
    rows = []
    for a, b in itertools.combinations(LLM_ORDER, 2):
        x, y = wide[a].to_numpy(), wide[b].to_numpy()
        try:
            w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            stat, p = w.statistic, w.pvalue
        except ValueError:
            stat, p = np.nan, 1.0
        rows.append({"Outcome": outcome, "LLM_A": a, "LLM_B": b, "Statistic": stat, "p_raw": p})
    pair = pd.DataFrame(rows)
    pair["p_holm"] = holm_adjust(pair["p_raw"])
    pair["Significance"] = pair["p_holm"].map(stars)
    return omnibus, pair


def independent_binary_posthoc(df: pd.DataFrame, outcome: str, group: str, per_llm: bool = False) -> pd.DataFrame:
    """Pairwise Fisher exact tests between group levels with Holm adjustment."""
    scopes = [("Overall", df)] if not per_llm else [(str(k), g) for k, g in df.groupby("LLM", observed=True)]
    rows = []
    for scope, g in scopes:
        levels = [x for x in g[group].dropna().unique()]
        for a, b in itertools.combinations(levels, 2):
            d = g[g[group].isin([a, b])]
            tab = pd.crosstab(d[group], d[outcome]).reindex(index=[a, b], columns=[0, 1], fill_value=0)
            if tab.shape != (2, 2):
                continue
            odds, p = stats.fisher_exact(tab.to_numpy())
            rows.append({
                "Scope": scope, "Outcome": outcome, "Group": group,
                "Level_A": str(a), "Level_B": str(b),
                "Odds_Ratio_unadjusted": odds, "p_raw": p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for _, idx in out.groupby(["Scope", "Outcome", "Group"]).groups.items():
            out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_raw"])
        out["Significance"] = out["p_holm"].map(stars)
    return out


def independent_continuous_test(df: pd.DataFrame, outcome: str, group: str, per_llm: bool = False, posthoc: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = [("Overall", df)] if not per_llm else [(str(k), g) for k, g in df.groupby("LLM", observed=True)]
    omnibus_rows, post_rows = [], []
    for scope, g in scopes:
        levels = [x for x in g[group].dropna().unique()]
        arrays = [g.loc[g[group] == lev, outcome].dropna().to_numpy() for lev in levels]
        arrays = [a for a in arrays if len(a) > 0]
        if len(arrays) < 2:
            continue
        if len(arrays) == 2:
            res = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
            omnibus_rows.append({"Scope": scope, "Outcome": outcome, "Group": group, "Test": "Mann-Whitney U", "Statistic": res.statistic, "p": res.pvalue})
        else:
            res = stats.kruskal(*arrays)
            omnibus_rows.append({"Scope": scope, "Outcome": outcome, "Group": group, "Test": "Kruskal-Wallis", "Statistic": res.statistic, "df": len(arrays)-1, "p": res.pvalue})
            if posthoc:
                for a, b in itertools.combinations(levels, 2):
                    xa = g.loc[g[group] == a, outcome].dropna()
                    xb = g.loc[g[group] == b, outcome].dropna()
                    if len(xa) and len(xb):
                        r = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                        post_rows.append({"Scope": scope, "Outcome": outcome, "Group": group, "Level_A": str(a), "Level_B": str(b), "Statistic": r.statistic, "p_raw": r.pvalue})
    omnibus = pd.DataFrame(omnibus_rows)
    post = pd.DataFrame(post_rows)
    if not post.empty:
        # Holm adjustment within scope/outcome/group analysis family.
        post["p_holm"] = np.nan
        for _, idx in post.groupby(["Scope", "Outcome", "Group"]).groups.items():
            post.loc[idx, "p_holm"] = holm_adjust(post.loc[idx, "p_raw"])
        post["Significance"] = post["p_holm"].map(stars)
    if per_llm and not omnibus.empty:
        omnibus["p_holm_across_LLMs"] = holm_adjust(omnibus["p"])
    return omnibus, post

# -----------------------------------------------------------------------------
# GEE analyses
# -----------------------------------------------------------------------------

def fit_gee(df: pd.DataFrame, formula: str, family, groups: str = "Case_ID"):
    d = df.copy()
    model = smf.gee(formula=formula, groups=d[groups], data=d, family=family, cov_struct=Exchangeable())
    return model.fit()


def gee_result_table(result, exponentiate: bool = False) -> pd.DataFrame:
    ci = result.conf_int()
    rows = []
    for term in result.params.index:
        est = result.params[term]
        lo, hi = ci.loc[term]
        if exponentiate:
            est_out, lo_out, hi_out = np.exp(est), np.exp(lo), np.exp(hi)
        else:
            est_out, lo_out, hi_out = est, lo, hi
        rows.append({
            "Term": term,
            "Estimate": est_out,
            "CI95_Lower": lo_out,
            "CI95_Upper": hi_out,
            "SE": result.bse[term],
            "z": result.tvalues[term],
            "p": result.pvalues[term],
            "Scale": "OR" if exponentiate else "coefficient",
        })
    return pd.DataFrame(rows)


def treatment(var: str, ref: str) -> str:
    return f"C({var}, Treatment(reference={ref!r}))"


def pooled_gee_characteristics(df: pd.DataFrame, outcome: str, kind: str) -> dict[str, pd.DataFrame]:
    """Separate pooled GEE models by each case characteristic, adjusted for LLM identity."""
    out = {}
    family = Gaussian() if kind == "gaussian" else Binomial()
    exp = kind == "binomial"
    for group, ref in [
        ("Disease_Category", "Neoplastic"),
        ("Rarity", "Common"),
        ("Diagnostic_Efficiency", "Fully sufficient"),
    ]:
        formula = f"{outcome} ~ {treatment('LLM','LLM1')} + {treatment(group,ref)}"
        res = fit_gee(df, formula, family)
        tab = gee_result_table(res, exponentiate=exp)
        tab.insert(0, "Model", f"{outcome} by {group}")
        out[group] = tab
    return out


def gee_correct_vs_incorrect(df: pd.DataFrame, outcome: str, kind: str = "gaussian") -> pd.DataFrame:
    d = df[df["Diagnosis_Correct"].isin([0, 2])].copy()
    d["Correct_vs_Incorrect"] = np.where(d["Diagnosis_Correct"] == 2, "Correct", "Incorrect")
    family = Gaussian() if kind == "gaussian" else Binomial()
    formula = f"{outcome} ~ {treatment('LLM','LLM1')} + C(Correct_vs_Incorrect, Treatment(reference='Correct'))"
    res = fit_gee(d, formula, family)
    return gee_result_table(res, exponentiate=(kind == "binomial"))


def gee_subtype_impact(df: pd.DataFrame, subtype_group: str, high_impact: bool = False) -> pd.DataFrame:
    if subtype_group == "errors":
        predictors = [f"Any_{c}" for c in ERROR_SUBTYPES]
    elif subtype_group == "hallucinations":
        predictors = [f"Any_{c}" for c in HALLUCINATION_SUBTYPES]
    else:
        raise ValueError("subtype_group must be 'errors' or 'hallucinations'")
    outcome = "High_Clinical_Impact" if high_impact else "Clinical_Impact"
    formula = outcome + " ~ " + treatment("LLM", "LLM1") + " + " + " + ".join(predictors)
    res = fit_gee(df, formula, Binomial() if high_impact else Gaussian())
    tab = gee_result_table(res, exponentiate=high_impact)
    tab.insert(0, "Subtype_Group", subtype_group)
    tab.insert(1, "Outcome", outcome)
    return tab


def gee_llm_binary_contrasts(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    formula = f"{outcome} ~ {treatment('LLM','LLM1')}"
    res = fit_gee(df, formula, Binomial())
    params = res.params
    cov = res.cov_params()

    # Overall Wald test of the three LLM coefficients.
    terms = [t for t in params.index if t.startswith("C(LLM")]
    if terms:
        b = params[terms].to_numpy()
        V = cov.loc[terms, terms].to_numpy()
        stat = float(b.T @ np.linalg.pinv(V) @ b)
        p = float(stats.chi2.sf(stat, len(terms)))
    else:
        stat, p = np.nan, np.nan
    overall = pd.DataFrame([{"Outcome": outcome, "Wald_chi2": stat, "df": len(terms), "p": p}])

    def coef_vector(llm: str) -> np.ndarray:
        v = np.zeros(len(params))
        if llm != "LLM1":
            target = [t for t in params.index if f"[T.{llm}]" in t]
            if not target:
                raise KeyError(f"Could not find coefficient for {llm}")
            v[params.index.get_loc(target[0])] = 1.0
        return v

    rows = []
    for a, b in itertools.combinations(LLM_ORDER, 2):
        # B vs A, in the ordered pair.
        c = coef_vector(b) - coef_vector(a)
        est = float(c @ params.to_numpy())
        se = float(np.sqrt(c @ cov.to_numpy() @ c))
        z = est / se if se > 0 else np.nan
        p_raw = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append({
            "Outcome": outcome,
            "Contrast": f"{b} vs {a}",
            "OR": np.exp(est),
            "CI95_Lower": np.exp(est - 1.96*se),
            "CI95_Upper": np.exp(est + 1.96*se),
            "z": z,
            "p_raw": p_raw,
        })
    pair = pd.DataFrame(rows)
    pair["p_holm"] = holm_adjust(pair["p_raw"])
    pair["Significance"] = pair["p_holm"].map(stars)
    return overall, pair

# -----------------------------------------------------------------------------
# Ordinal regression
# -----------------------------------------------------------------------------

def ordinal_burden_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for llm in LLM_ORDER:
        d = df[df["LLM"] == llm].copy()
        y = d["Diagnosis_Correct"].astype(int)
        for burden in ["Error_Burden", "Hallucination_Burden"]:
            x = d[[burden]].astype(float)
            # OrderedModel adds thresholds internally; no explicit intercept should be supplied.
            model = OrderedModel(y, x, distr="logit")
            res = model.fit(method="bfgs", disp=False)
            beta = float(res.params[burden])
            se = float(res.bse[burden])
            rows.append({
                "LLM": llm,
                "Predictor": burden,
                "OR_per_point": np.exp(beta),
                "CI95_Lower": np.exp(beta - 1.96*se),
                "CI95_Upper": np.exp(beta + 1.96*se),
                "Coefficient": beta,
                "SE": se,
                "z": beta/se,
                "p": 2*stats.norm.sf(abs(beta/se)),
            })
    out = pd.DataFrame(rows)
    out["p_holm_within_burden"] = np.nan
    for burden, idx in out.groupby("Predictor").groups.items():
        out.loc[idx, "p_holm_within_burden"] = holm_adjust(out.loc[idx, "p"])
    return out

# -----------------------------------------------------------------------------
# Bayesian mixed-effects logistic regression
# -----------------------------------------------------------------------------

def bayes_formula(outcome: str) -> str:
    base = (
        f"{outcome} ~ {treatment('LLM','LLM1')} + "
        f"{treatment('Disease_Category','Neoplastic')} + "
        f"{treatment('Rarity','Common')} + "
        f"{treatment('Diagnostic_Efficiency','Fully sufficient')}"
    )
    if outcome in {"High_Clinical_Impact", "Low_Overall_Score"}:
        base += " + C(Diagnosis_Correct, Treatment(reference=2)) + Error_Burden + Hallucination_Burden"
    return base


def fit_bayes_mixed_logit(df: pd.DataFrame, outcome: str):
    formula = bayes_formula(outcome)
    vc = {
        "Case": "0 + C(Case_ID)",
        "Organ": "0 + C(Organ)",
    }
    model = BinomialBayesMixedGLM.from_formula(formula, vc, df, vcp_p=1, fe_p=2)
    result = model.fit_vb(scale_fe=False, verbose=False)
    return model, result


def bayes_fixed_effect_table(model, result, outcome: str) -> pd.DataFrame:
    names = model.exog_names
    means = np.asarray(result.fe_mean)
    sds = np.asarray(result.fe_sd)
    rows = []
    for name, mean, sd in zip(names, means, sds):
        rows.append({
            "Outcome": outcome,
            "Term": name,
            "Log_Odds_Mean": mean,
            "Posterior_SD": sd,
            "aOR": np.exp(mean),
            "CrI95_Lower": np.exp(mean - 1.96*sd),
            "CrI95_Upper": np.exp(mean + 1.96*sd),
            "CrI_excludes_1": bool((mean - 1.96*sd > 0) or (mean + 1.96*sd < 0)),
        })
    return pd.DataFrame(rows)


def bayes_llm_contrasts(model, result, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Approximate joint/pairwise Wald tests from the VB fixed-effect marginal posterior.

    BinomialBayesMixedGLM.fit_vb reports marginal means and SDs. The variational posterior
    is factorized, so this implementation uses a diagonal covariance matrix for the fixed effects,
    matching the approximation used for the manuscript's LLM contrast calculations.
    """
    names = list(model.exog_names)
    mu = np.asarray(result.fe_mean)
    sd = np.asarray(result.fe_sd)
    V = np.diag(sd**2)

    def llm_term(llm: str) -> int | None:
        if llm == "LLM1": return None
        matches = [i for i, n in enumerate(names) if n.startswith("C(LLM") and f"[T.{llm}]" in n]
        if not matches:
            raise KeyError(f"Cannot find fixed effect for {llm} in {names}")
        return matches[0]

    idx = [llm_term(x) for x in ["LLM2", "LLM3", "LLM4"]]
    idx = [i for i in idx if i is not None]
    b = mu[idx]
    VV = V[np.ix_(idx, idx)]
    stat = float(b.T @ np.linalg.pinv(VV) @ b)
    overall = pd.DataFrame([{
        "Outcome": outcome,
        "Test": "Approximate joint Wald test for LLM identity",
        "Chi2": stat,
        "df": 3,
        "p": float(stats.chi2.sf(stat, 3)),
    }])

    def vector(llm: str) -> np.ndarray:
        v = np.zeros(len(mu))
        i = llm_term(llm)
        if i is not None: v[i] = 1.0
        return v

    rows = []
    # Explicit manuscript contrast order.
    pairs = [("LLM2","LLM1"),("LLM3","LLM1"),("LLM4","LLM1"),("LLM3","LLM2"),("LLM4","LLM2"),("LLM4","LLM3")]
    for numerator, denominator in pairs:
        c = vector(numerator) - vector(denominator)
        est = float(c @ mu)
        se = float(np.sqrt(c @ V @ c))
        z = est / se if se > 0 else np.nan
        p = 2*stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append({
            "Outcome": outcome,
            "Contrast": f"{numerator} vs {denominator}",
            "aOR": np.exp(est),
            "CrI95_Lower": np.exp(est - 1.96*se),
            "CrI95_Upper": np.exp(est + 1.96*se),
            "Approx_z": z,
            "p_raw": p,
        })
    pair = pd.DataFrame(rows)
    pair["p_holm"] = holm_adjust(pair["p_raw"])
    pair["Significance"] = pair["p_holm"].map(stars)
    return overall, pair

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def save_svg(fig, path: Path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_grouped_proportions(summary: pd.DataFrame, x: str, hue: str | None, title: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue is None:
        s = summary.copy()
        labels = s[x].astype(str).tolist()
        vals = s["Percent"].to_numpy()
        yerr = np.vstack([(s["Proportion"]-s["CI95_Lower"])*100, (s["CI95_Upper"]-s["Proportion"])*100])
        ax.bar(np.arange(len(vals)), vals)
        ax.errorbar(np.arange(len(vals)), vals, yerr=yerr, fmt="none", capsize=3)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=30, ha="right")
    else:
        xs = [xv for xv in summary[x].dropna().unique()]
        hs = [hv for hv in summary[hue].dropna().unique()]
        width = 0.8/max(len(hs), 1)
        pos = np.arange(len(xs))
        for j, hv in enumerate(hs):
            s = summary[summary[hue] == hv].set_index(x).reindex(xs)
            vals = s["Percent"].to_numpy()
            ax.bar(pos - 0.4 + width/2 + j*width, vals, width=width, label=str(hv))
        ax.set_xticks(pos, [str(z) for z in xs], rotation=25, ha="right")
        ax.legend(frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    save_svg(fig, path)


def plot_mean_points(summary: pd.DataFrame, x: str, hue: str | None, title: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue is None:
        s = summary.copy()
        pos = np.arange(len(s))
        ax.errorbar(pos, s["Mean"], yerr=[s["Mean"]-s["CI95_Lower"], s["CI95_Upper"]-s["Mean"]], fmt="o", capsize=3)
        ax.set_xticks(pos, s[x].astype(str), rotation=25, ha="right")
    else:
        xs = [xv for xv in summary[x].dropna().unique()]
        hs = [hv for hv in summary[hue].dropna().unique()]
        pos = np.arange(len(xs))
        offsets = np.linspace(-0.15, 0.15, len(hs)) if len(hs)>1 else [0]
        for off, hv in zip(offsets, hs):
            s = summary[summary[hue] == hv].set_index(x).reindex(xs)
            ax.errorbar(pos+off, s["Mean"], yerr=[s["Mean"]-s["CI95_Lower"], s["CI95_Upper"]-s["Mean"]], fmt="o", capsize=3, label=str(hv))
        ax.set_xticks(pos, [str(z) for z in xs], rotation=25, ha="right")
        ax.legend(frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    save_svg(fig, path)


def generate_summary_figures(df: pd.DataFrame, paths: AnalysisPaths):
    # Figure 1-style diagnostic accuracy summaries.
    for group, label in [
        ("LLM", "LLM"),
        ("Disease_Category", "Disease category"),
        ("Rarity", "Disease rarity"),
        ("Diagnostic_Efficiency", "Diagnostic efficiency"),
        ("Organ", "Organ"),
    ]:
        s = binary_summary(df, "Strict_Correct", [group])
        plot_grouped_proportions(s, group, None, f"Strict diagnostic accuracy by {label.lower()}", "Strict accuracy (%)", paths.figures / f"accuracy_by_{clean_name(label)}.svg")

    # Error/hallucination prevalence by LLM.
    prev = pd.concat([binary_summary(df, o, ["LLM"]) for o in ["Any_Error", "Any_Hallucination"]], ignore_index=True)
    prev = prev.rename(columns={"Outcome":"Failure_Type"})
    plot_grouped_proportions(prev, "LLM", "Failure_Type", "Errors and hallucinations by LLM", "Outputs (%)", paths.figures / "errors_hallucinations_by_llm.svg")

    # Burden by LLM.
    burden = pd.concat([continuous_summary(df, o, ["LLM"]) for o in ["Error_Burden", "Hallucination_Burden"]], ignore_index=True)
    burden = burden.rename(columns={"Outcome":"Burden_Type"})
    plot_mean_points(burden, "LLM", "Burden_Type", "Error and hallucination burden by LLM", "Mean burden", paths.figures / "burden_by_llm.svg")

    # Clinical impact and overall score.
    for outcome, ylabel in [("Clinical_Impact","Mean clinical impact"),("Overall_Score","Mean overall score")]:
        s = continuous_summary(df, outcome, ["LLM"])
        plot_mean_points(s, "LLM", None, outcome.replace("_"," "), ylabel, paths.figures / f"{clean_name(outcome)}_by_llm.svg")

    # Safety profile.
    safety = pd.concat([binary_summary(df, o, ["LLM"]) for o in ["Safe_Correct", "Dangerous_Wrong"]], ignore_index=True).rename(columns={"Outcome":"Safety_Class"})
    plot_grouped_proportions(safety, "LLM", "Safety_Class", "Clinical safety profile by LLM", "Outputs (%)", paths.figures / "safety_profile_by_llm.svg")

# -----------------------------------------------------------------------------
# Analysis orchestration
# -----------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, paths: AnalysisPaths, skip_bayesian: bool = False, skip_figures: bool = False):
    # Save canonical analysis dataset without altering raw input.
    df.to_csv(paths.root / "analysis_dataset_derived.csv", index=False)

    # 1. Dataset characteristics.
    save_df(dataset_characteristics(df), paths.tables / "table_dataset_characteristics.csv")

    # 2. Diagnostic performance.
    diag_summaries = [binary_summary(df, "Strict_Correct"), binary_summary(df, "Partial_or_Correct")]
    for outcome in ["Strict_Correct", "Partial_or_Correct"]:
        for by in [["LLM"], ["Disease_Category"], ["Rarity"], ["Diagnostic_Efficiency"], ["Organ"], ["LLM","Disease_Category"], ["LLM","Rarity"], ["LLM","Diagnostic_Efficiency"], ["LLM","Organ"]]:
            diag_summaries.append(binary_summary(df, outcome, by))
    save_df(pd.concat(diag_summaries, ignore_index=True), paths.tables / "diagnostic_accuracy_summaries.csv")
    coverage_summary, coverage_cases = case_level_coverage(df)
    save_df(coverage_summary, paths.tables / "case_level_coverage_summary.csv")
    save_df(coverage_cases, paths.tables / "case_level_coverage_by_case.csv")

    omnibus_list, pair_list = [], []
    for outcome in ["Strict_Correct", "Partial_or_Correct"]:
        om, pa = paired_binary_llm_tests(df, outcome)
        omnibus_list.append(om); pair_list.append(pa)
        for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
            save_df(independent_binary_test(df, outcome, group, per_llm=False), paths.tables / f"{clean_name(outcome)}_{clean_name(group)}_overall_test.csv")
            save_df(independent_binary_test(df, outcome, group, per_llm=True), paths.tables / f"{clean_name(outcome)}_{clean_name(group)}_per_llm_test.csv")
            if group in {"Rarity", "Diagnostic_Efficiency", "Organ"}:
                save_df(independent_binary_posthoc(df, outcome, group, per_llm=False), paths.tables / f"{clean_name(outcome)}_{clean_name(group)}_overall_posthoc.csv")
                save_df(independent_binary_posthoc(df, outcome, group, per_llm=True), paths.tables / f"{clean_name(outcome)}_{clean_name(group)}_per_llm_posthoc.csv")
    save_df(pd.concat(omnibus_list, ignore_index=True), paths.tables / "paired_binary_llm_omnibus_diagnostic.csv")
    save_df(pd.concat(pair_list, ignore_index=True), paths.tables / "paired_binary_llm_pairwise_diagnostic.csv")

    # 3. Errors and hallucinations.
    binary_failures = ["Any_Error", "Any_Hallucination", "Any_Failure"] + [f"Any_{c}" for c in ALL_SUBTYPES]
    prev_tables = []
    for outcome in binary_failures:
        prev_tables += [binary_summary(df, outcome), binary_summary(df, outcome, ["LLM"])]
        for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ", "Diagnosis_Correct"]:
            prev_tables.append(binary_summary(df, outcome, [group]))
            prev_tables.append(binary_summary(df, outcome, ["LLM", group]))
        om, pa = paired_binary_llm_tests(df, outcome)
        save_df(om, paths.tables / f"paired_{clean_name(outcome)}_cochranq.csv")
        save_df(pa, paths.tables / f"paired_{clean_name(outcome)}_mcnemar.csv")
    save_df(pd.concat(prev_tables, ignore_index=True, sort=False), paths.tables / "error_hallucination_prevalence_all_summaries.csv")

    # Correct vs incorrect (strict only) prevalence.
    strict = df[df["Diagnosis_Correct"].isin([0,2])].copy()
    save_df(pd.concat([binary_summary(strict, o, ["Diagnosis_Correct"]) for o in ["Any_Error","Any_Hallucination","Any_Failure"]], ignore_index=True), paths.tables / "failures_correct_vs_incorrect.csv")

    # GEE binary analyses across characteristics.
    for outcome in ["Any_Error", "Any_Hallucination"]:
        for group, ref in [("Disease_Category","Neoplastic"),("Rarity","Common"),("Diagnostic_Efficiency","Fully sufficient")]:
            formula = f"{outcome} ~ {treatment('LLM','LLM1')} + {treatment(group,ref)}"
            res = fit_gee(df, formula, Binomial())
            save_df(gee_result_table(res, exponentiate=True), paths.models / f"gee_{clean_name(outcome)}_by_{clean_name(group)}.csv")
        save_df(gee_correct_vs_incorrect(df, outcome, kind="binomial"), paths.models / f"gee_{clean_name(outcome)}_correct_vs_incorrect.csv")

    # Clinically relevant correctness+failure pattern comparisons across LLMs.
    for outcome in ["Correct_with_Any_Error", "Correct_with_Any_Hallucination", "Correct_with_Any_Failure", "Incorrect_with_Any_Error", "Incorrect_with_Any_Hallucination", "Incorrect_with_Any_Failure"]:
        om, pa = gee_llm_binary_contrasts(df, outcome)
        save_df(om, paths.models / f"gee_llm_{clean_name(outcome)}_overall.csv")
        save_df(pa, paths.models / f"gee_llm_{clean_name(outcome)}_pairwise.csv")

    # 4. Burden analyses.
    burden_summaries = []
    for outcome in ["Error_Burden", "Hallucination_Burden"]:
        burden_summaries += [continuous_summary(df, outcome), continuous_summary(df, outcome, ["LLM"])]
        for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
            burden_summaries.append(continuous_summary(df, outcome, [group]))
            burden_summaries.append(continuous_summary(df, outcome, ["LLM", group]))
        burden_summaries.append(continuous_summary(strict, outcome, ["Diagnosis_Correct"]))
        burden_summaries.append(continuous_summary(strict, outcome, ["LLM","Diagnosis_Correct"]))

        om, pa = paired_continuous_llm_tests(df, outcome)
        save_df(om, paths.tables / f"{clean_name(outcome)}_friedman.csv")
        save_df(pa, paths.tables / f"{clean_name(outcome)}_wilcoxon_pairwise.csv")
        for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
            om2, post2 = independent_continuous_test(df, outcome, group, per_llm=True)
            save_df(om2, paths.tables / f"{clean_name(outcome)}_{clean_name(group)}_per_llm_omnibus.csv")
            save_df(post2, paths.tables / f"{clean_name(outcome)}_{clean_name(group)}_per_llm_posthoc.csv")
        # Pooled GEE adjusted for LLM identity.
        for group, tab in pooled_gee_characteristics(df, outcome, "gaussian").items():
            save_df(tab, paths.models / f"gee_{clean_name(outcome)}_by_{clean_name(group)}.csv")
        save_df(gee_correct_vs_incorrect(df, outcome, "gaussian"), paths.models / f"gee_{clean_name(outcome)}_correct_vs_incorrect.csv")
    save_df(pd.concat(burden_summaries, ignore_index=True, sort=False), paths.tables / "burden_all_summaries.csv")
    save_df(ordinal_burden_models(df), paths.models / "ordinal_logit_burden_vs_diagnostic_correctness.csv")

    # 5. Clinical impact analyses.
    impact_summaries = [continuous_summary(df, "Clinical_Impact"), continuous_summary(df, "Clinical_Impact", ["LLM"])]
    for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
        impact_summaries.append(continuous_summary(df, "Clinical_Impact", [group]))
        impact_summaries.append(continuous_summary(df, "Clinical_Impact", ["LLM",group]))
    save_df(pd.concat(impact_summaries, ignore_index=True, sort=False), paths.tables / "clinical_impact_summaries.csv")
    om, pa = paired_continuous_llm_tests(df, "Clinical_Impact")
    save_df(om, paths.tables / "clinical_impact_friedman.csv")
    save_df(pa, paths.tables / "clinical_impact_wilcoxon_pairwise.csv")
    for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
        om2, post2 = independent_continuous_test(df, "Clinical_Impact", group, per_llm=False)
        save_df(om2, paths.tables / f"clinical_impact_{clean_name(group)}_omnibus.csv")
        save_df(post2, paths.tables / f"clinical_impact_{clean_name(group)}_posthoc.csv")

    for subtype_group in ["errors", "hallucinations"]:
        save_df(gee_subtype_impact(df, subtype_group, high_impact=False), paths.models / f"gee_{subtype_group}_subtypes_mean_clinical_impact.csv")
        save_df(gee_subtype_impact(df, subtype_group, high_impact=True), paths.models / f"gee_{subtype_group}_subtypes_high_clinical_impact.csv")

    # Safety classes.
    safety_summary = pd.concat([binary_summary(df, x) for x in ["Safe_Correct","Dangerous_Wrong"]] + [binary_summary(df, x, ["LLM"]) for x in ["Safe_Correct","Dangerous_Wrong"]], ignore_index=True, sort=False)
    save_df(safety_summary, paths.tables / "safety_classifications.csv")
    for outcome in ["Safe_Correct", "Dangerous_Wrong"]:
        om, pa = paired_binary_llm_tests(df, outcome)
        save_df(om, paths.tables / f"{clean_name(outcome)}_cochranq.csv")
        save_df(pa, paths.tables / f"{clean_name(outcome)}_mcnemar_pairwise.csv")

    # Additional denominators reported in Results.
    denominator_rows = []
    for diag_value, diag_label in [(2,"Strictly correct"),(0,"Incorrect")]:
        d = df[df["Diagnosis_Correct"] == diag_value]
        for outcome in ["Any_Error","Any_Hallucination","Any_Failure","Safe_Correct","Dangerous_Wrong"]:
            k = int(d[outcome].sum()); n=len(d)
            denominator_rows.append({"Diagnosis_Group":diag_label,"Outcome":outcome,"Events":k,"N":n,"Percent":100*k/n if n else np.nan})
    save_df(pd.DataFrame(denominator_rows), paths.tables / "failure_and_safety_within_diagnostic_groups.csv")

    # 6. Overall performance score.
    overall_summaries = [continuous_summary(df, "Overall_Score"), continuous_summary(df, "Overall_Score", ["LLM"])]
    for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
        overall_summaries.append(continuous_summary(df, "Overall_Score", [group]))
        overall_summaries.append(continuous_summary(df, "Overall_Score", ["LLM", group]))
    save_df(pd.concat(overall_summaries, ignore_index=True, sort=False), paths.tables / "overall_score_summaries.csv")
    om, pa = paired_continuous_llm_tests(df, "Overall_Score")
    save_df(om, paths.tables / "overall_score_friedman.csv")
    save_df(pa, paths.tables / "overall_score_wilcoxon_pairwise.csv")
    for group in ["Disease_Category", "Rarity", "Diagnostic_Efficiency", "Organ"]:
        om2, post2 = independent_continuous_test(df, "Overall_Score", group, per_llm=False)
        save_df(om2, paths.tables / f"overall_score_{clean_name(group)}_omnibus.csv")
        save_df(post2, paths.tables / f"overall_score_{clean_name(group)}_posthoc.csv")

    # 7. Five Bayesian mixed-effects logistic models.
    if not skip_bayesian:
        bayes_outcomes = ["Incorrect", "Any_Error", "Any_Hallucination", "High_Clinical_Impact", "Low_Overall_Score"]
        all_fixed, all_joint, all_pair = [], [], []
        for outcome in bayes_outcomes:
            model, result = fit_bayes_mixed_logit(df, outcome)
            fixed = bayes_fixed_effect_table(model, result, outcome)
            joint, pair = bayes_llm_contrasts(model, result, outcome)
            save_df(fixed, paths.models / f"bayesian_mixed_logit_{clean_name(outcome)}_fixed_effects.csv")
            save_df(joint, paths.models / f"bayesian_mixed_logit_{clean_name(outcome)}_llm_joint_wald.csv")
            save_df(pair, paths.models / f"bayesian_mixed_logit_{clean_name(outcome)}_llm_pairwise.csv")
            all_fixed.append(fixed); all_joint.append(joint); all_pair.append(pair)
            # Save key fit metadata.
            meta = {
                "outcome": outcome,
                "formula": bayes_formula(outcome),
                "random_effects": {"Case":"0 + C(Case_ID)", "Organ":"0 + C(Organ)"},
                "vcp_p": 1,
                "fe_p": 2,
                "fit": "variational Bayes (fit_vb)",
            }
            write_json(meta, paths.models / f"bayesian_mixed_logit_{clean_name(outcome)}_metadata.json")
        save_df(pd.concat(all_fixed, ignore_index=True), paths.models / "bayesian_all_fixed_effects.csv")
        save_df(pd.concat(all_joint, ignore_index=True), paths.models / "bayesian_all_llm_joint_wald.csv")
        save_df(pd.concat(all_pair, ignore_index=True), paths.models / "bayesian_all_llm_pairwise_contrasts.csv")

    # 8. Optional response-length summaries.
    if "Response_Word_Count" in df.columns:
        save_df(continuous_summary(df, "Response_Word_Count", ["LLM"]), paths.tables / "exploratory_response_word_count_by_llm.csv")
        for outcome in ["Any_Error", "Any_Hallucination", "Strict_Correct"]:
            formula = f"{outcome} ~ {treatment('LLM','LLM1')} + Response_Word_Count"
            res = fit_gee(df, formula, Binomial())
            save_df(gee_result_table(res, exponentiate=True), paths.models / f"exploratory_gee_{clean_name(outcome)}_response_length.csv")

    # 9. Figures.
    if not skip_figures:
        generate_summary_figures(df, paths)

    # 10. Session information and checks.
    checks = {
        "rows": len(df),
        "cases": int(df["Case_ID"].nunique()),
        "organs": int(df["Organ"].nunique()),
        "llms": [str(x) for x in df["LLM"].dropna().unique()],
        "case_x_llm_unique": int(df[["Case_ID","LLM"]].drop_duplicates().shape[0]),
        "duplicate_case_llm_rows": int(df.duplicated(["Case_ID","LLM"]).sum()),
        "missing_required_values": {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS},
    }
    write_json(checks, paths.logs / "data_checks.json")

    session = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"pandas: {pd.__version__}",
        f"numpy: {np.__version__}",
        f"scipy: {scipy.__version__}",
        f"statsmodels: {statsmodels.__version__}",
        f"matplotlib: {plt.matplotlib.__version__}",
    ]
    (paths.logs / "session_info.txt").write_text("\n".join(session) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the LLM pathology error/hallucination study analyses.")
    p.add_argument("--input", required=True, help="Long-format .xlsx/.xls/.csv/.tsv input dataset")
    p.add_argument("--sheet", default=None, help="Excel sheet name (default: first sheet)")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--strict-shape", action="store_true", help="Require exactly 153 cases and 612 outputs")
    p.add_argument("--skip-bayesian", action="store_true", help="Skip Bayesian mixed-effects models")
    p.add_argument("--skip-figures", action="store_true", help="Skip SVG figure generation")
    return p.parse_args()


def main():
    args = parse_args()
    paths = AnalysisPaths.create(args.output)
    raw = load_input(args.input, args.sheet)
    df = validate_and_derive(raw, strict_shape=args.strict_shape)
    run_analysis(df, paths, skip_bayesian=args.skip_bayesian, skip_figures=args.skip_figures)
    print(f"Analysis complete. Outputs written to: {paths.root.resolve()}")


if __name__ == "__main__":
    main()
