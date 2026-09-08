#!/usr/bin/env python3
"""Create a synthetic 153-case x 4-LLM dataset for testing the public analysis pipeline."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

LLMS = ["LLM1", "LLM2", "LLM3", "LLM4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/synthetic_example.csv")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rarities = ["Common"]*67 + ["Sporadic"]*63 + ["Rare"]*23
    efficiencies = ["Fully sufficient"]*55 + ["Additional support desirable"]*40 + ["Additional support required"]*58
    rows=[]
    for i in range(153):
        case=f"SYN{i+1:03d}"
        organ=f"SyntheticOrgan{(i%20)+1:02d}"
        disease="Neoplastic" if i < 122 else "Non-neoplastic"
        for llm_i,llm in enumerate(LLMS):
            # Deliberately synthetic; these probabilities do not encode study results.
            p_correct=max(0.15,min(0.75,0.52 - 0.06*(llm_i==2) + 0.04*(llm_i==3)))
            diagnosis=int(rng.choice([0,1,2], p=[1-p_correct-0.12,0.12,p_correct]))
            base_fail=0.4 + 0.25*(diagnosis==0)
            errs=[int(rng.binomial(3,min(0.9,base_fail))) for _ in range(3)]
            halls=[int(rng.binomial(3,min(0.85,base_fail-0.05))) for _ in range(3)]
            impact=int(rng.integers(0,5))
            overall=int(rng.integers(1,6))
            rows.append([case,organ,llm,disease,rarities[i],efficiencies[i],diagnosis,*errs,*halls,impact,overall])
    cols=["Case_ID","Organ","LLM","Disease_Category","Rarity","Diagnostic_Efficiency","Diagnosis_Correct",
          "Omission","Misinterpretation","Internal_Inconsistency","Fabricated_Feature","Unsupported_Inference",
          "Instruction_Hallucination","Clinical_Impact","Overall_Score"]
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=cols).to_csv(out,index=False)
    print(f"Wrote {len(rows)} synthetic rows to {out}")

if __name__ == "__main__":
    main()
