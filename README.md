# Plurilateral Trade Agreements: A Complementary Margin to Preferential Liberalization

**Lasha Chochua** (TSU ISET), **James Lake** (University of Tennessee), **Gerald Willmann** (Uni Bielefeld, IfW Kiel)

## Overview

This repository contains all replication files for the paper. The paper develops a 
three-country model of endogenous trade agreement formation with farsighted governments 
and studies how plurilateral MFN-based agreements interact with preferential trade 
agreements (PTAs). We consider three environments: symmetric countries, exclusion 
incentives (two small countries and one large country), and free-riding incentives 
(one small country and two large countries).

## Repository Structure
```
CLW-paper-supplement/
├── README.md
├── requirements.txt
├── lcs_python_files/
│   └── Python scripts that compute the Largest Consistent Set (LCS)
│       across the (α, σ) parameter space for each of the three
│       country asymmetry cases.
└── utilities_trade_networks_mathematica_files/
    └── Mathematica notebooks that solve the welfare maximization
        problem and compute optimal tariffs across all 31 trade
        agreement networks.
```

## Requirements

### Python
- Python 3.11.3
- See `requirements.txt` for required packages

Install dependencies with:
```bash
pip install -r requirements.txt
```

### Mathematica
- Mathematica 12.3 or later

## Replication

**Step 1.** Run the Mathematica notebooks in
`utilities_trade_networks_mathematica_files/` to compute optimal tariffs
and welfare levels across all trade agreement networks. The notebooks
generate Excel (.xlsx) output files.

**Step 2.** Place all generated Excel files together with the Python scripts
from `lcs_python_files/` in the same folder before running the Python code.

**Step 3.** Run the Python scripts to compute the Largest Consistent Set
across the (α, σ) parameter space.

## Citation

If you use these files, please cite:

Chochua, L., Lake, J., and Willmann, G. "Plurilateral Trade Agreements: 
A Complementary Margin to Preferential Liberalization."