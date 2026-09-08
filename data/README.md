# Data dictionary

The analysis dataset is not included in this repository. The analysis code expects one row per LLM output.

For the final study dataset, each `Case_ID` should occur four times, once for each `LLM` (`LLM1`–`LLM4`).

## Required variables

- `Case_ID` — case identifier.
- `Organ` — organ/site.
- `LLM` — model identifier.
- `Disease_Category` — neoplastic/non-neoplastic.
- `Rarity` — common/sporadic/rare.
- `Diagnostic_Efficiency` — fully sufficient/additional support desirable/additional support required.
- `Diagnosis_Correct` — 0 incorrect, 1 partially correct, 2 correct.
- `Omission` — 0–3.
- `Misinterpretation` — 0–3.
- `Internal_Inconsistency` — 0–3.
- `Fabricated_Feature` — 0–3.
- `Unsupported_Inference` — 0–3.
- `Instruction_Hallucination` — 0–3.
- `Clinical_Impact` — 0–4.
- `Overall_Score` — 1–5.

## Optional variables

- `Institution`
- `Microscopic_Description`
- `Final_Diagnosis`

Do not commit patient identifiers, raw images, protected clinical information, or restricted study data to a public repository.
