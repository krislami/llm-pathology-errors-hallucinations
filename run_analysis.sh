#!/usr/bin/env bash
set -euo pipefail
python analysis.py --input data/LLM_Pathology_Analysis.xlsx --output results --strict-shape
