"""
TASK 1: Verify Hunter's data and RAG pipeline
Columns in the CSVs are UPPERCASE — we normalize to lowercase on load.
"""

import pandas as pd
import os
import sys

def load_csv(path):
    """Load CSV and normalize column names to lowercase."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df

# ──────────────────────────────────────────────
# SECTION 1A: EXPENDITURES CSV
# ──────────────────────────────────────────────
print("=" * 60)
print("TASK 1A: Verifying clean_county_expenditures.csv")
print("=" * 60)

exp_path = "data/clean/clean_county_expenditures.csv"
if not os.path.exists(exp_path):
    print(f"ERROR: File not found: {exp_path}")
    sys.exit(1)

df_exp = load_csv(exp_path)
print(f"\nShape: {df_exp.shape[0]:,} rows × {df_exp.shape[1]} columns")
print(f"\nColumn names (lowercased):\n  {list(df_exp.columns)}")
print(f"\nData types:\n{df_exp.dtypes}")

# Unique counties
unique_counties = sorted(df_exp["county"].dropna().unique())
print(f"\nUnique counties ({len(unique_counties)}): {unique_counties}")

# Unique years
years = sorted(df_exp["calendar_year"].dropna().unique().astype(int))
print(f"\nUnique calendar_year values: {years}")
most_recent_year = max(years)

# Level 1 categories
cats = sorted(df_exp["level_1_category"].dropna().unique())
print(f"\nUnique level_1_category values ({len(cats)}):")
for c in cats:
    print(f"  - {c}")

# Suffolk sample in most recent year
suffolk = df_exp[(df_exp["county"] == "Suffolk") & (df_exp["calendar_year"] == most_recent_year)]
print(f"\n--- Sample 10 rows: Suffolk County, {most_recent_year} ---")
print(suffolk.head(10).to_string(index=False))

# Total Suffolk spending
total_suffolk = suffolk["amount"].sum()
print(f"\nTotal Suffolk County spending ({most_recent_year}): ${total_suffolk:,.0f}")

# Null checks
print(f"\n--- Null/type checks ---")
null_amount = df_exp["amount"].isna().sum()
null_county = df_exp["county"].isna().sum()
is_numeric = pd.api.types.is_numeric_dtype(df_exp["amount"])
sample_vals = df_exp["amount"].dropna().head(5).tolist()

print(f"  Nulls in 'amount':  {null_amount:,}")
print(f"  Nulls in 'county':  {null_county:,}")
print(f"  'amount' dtype:     {df_exp['amount'].dtype}  (numeric: {is_numeric})")
print(f"  Sample amount values: {sample_vals}")

# ──────────────────────────────────────────────
# SECTION 1B: REVENUE CSV
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 1B: Verifying clean_county_revenue.csv")
print("=" * 60)

rev_path = "data/clean/clean_county_revenue.csv"
if not os.path.exists(rev_path):
    print(f"ERROR: File not found: {rev_path}")
else:
    df_rev = load_csv(rev_path)
    print(f"\nShape: {df_rev.shape[0]:,} rows × {df_rev.shape[1]} columns")
    print(f"\nColumn names (lowercased):\n  {list(df_rev.columns)}")
    print(f"\nData types:\n{df_rev.dtypes}")

    unique_rev_counties = sorted(df_rev["county"].dropna().unique())
    print(f"\nUnique counties ({len(unique_rev_counties)}): {unique_rev_counties}")

    rev_years = sorted(df_rev["calendar_year"].dropna().unique().astype(int))
    print(f"\nUnique calendar_year values: {rev_years}")

    rev_cats = sorted(df_rev["level_1_category"].dropna().unique())
    print(f"\nUnique level_1_category values ({len(rev_cats)}):")
    for c in rev_cats:
        print(f"  - {c}")

    null_amt_rev = df_rev["amount"].isna().sum()
    null_cty_rev = df_rev["county"].isna().sum()
    print(f"\nNulls in 'amount': {null_amt_rev:,}")
    print(f"Nulls in 'county': {null_cty_rev:,}")
    print(f"Amount is numeric: {pd.api.types.is_numeric_dtype(df_rev['amount'])}")

# ──────────────────────────────────────────────
# SECTION 1C: COLUMN REFERENCE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 1C: Reading column_reference.txt")
print("=" * 60)
col_ref_path = "data/reference/column_reference.txt"
with open(col_ref_path, "r") as f:
    col_ref = f.read()
print(col_ref)

# Verify it covers all actual columns
actual_cols = list(df_exp.columns)
col_ref_upper = col_ref.upper()
print("--- Coverage check (actual CSV columns vs reference) ---")
for col in actual_cols:
    covered = col.upper() in col_ref_upper
    print(f"  {'OK' if covered else 'MISSING'}: {col}")

# ──────────────────────────────────────────────
# SECTION 1D: BUILD_RAG.PY REVIEW
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 1D: Reading build_rag.py (root version)")
print("=" * 60)
with open("build_rag.py", "r") as f:
    rag_src = f.read()
print(rag_src)

print("\n--- build_rag.py checks ---")
checks = {
    "PersistentClient": "PersistentClient" in rag_src,
    "Reads from data/reference": "data/reference" in rag_src,
    "Chunks documents": "chunk" in rag_src.lower(),
    "Collection 'budget_docs'": "budget_docs" in rag_src,
    "No hardcoded absolute path": "county-budget-agent" not in rag_src,
}
for name, result in checks.items():
    print(f"  {'OK' if result else 'ISSUE'}: {name}")

print("\nAll Task 1 checks complete.")
print("Next: python build_rag.py  →  builds the ChromaDB vector store")
