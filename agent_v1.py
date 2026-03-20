"""
TASK 2: agent_v1.py — Basic single agent
Loads expenditure data + ChromaDB, takes a user question,
retrieves RAG context, and returns a professional briefing answer.

Usage:
    python agent_v1.py
    python agent_v1.py "Your custom question here"
"""

import os
import sys
import pandas as pd
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
EXP_CSV = "data/clean/clean_county_expenditures.csv"
MODEL = "claude-sonnet-4-6"
RAG_TOP_K = 5


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data():
    print("Loading expenditure data...")
    df = pd.read_csv(EXP_CSV)
    df.columns = df.columns.str.lower()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────
# CONNECT TO CHROMADB
# ──────────────────────────────────────────────────────────────
def get_chroma_collection():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("budget_docs")
    print(f"  Collection 'budget_docs': {collection.count()} chunks")
    return collection


# ──────────────────────────────────────────────────────────────
# RAG RETRIEVAL
# ──────────────────────────────────────────────────────────────
def retrieve_context(collection, question: str, top_k: int = RAG_TOP_K) -> str:
    results = collection.query(query_texts=[question], n_results=top_k)
    docs = results["documents"][0]
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Reference {i}]\n{doc[:600]}")
    return "\n\n".join(context_parts)


# ──────────────────────────────────────────────────────────────
# BUILD DATA METADATA (for the LLM prompt)
# ──────────────────────────────────────────────────────────────
def build_data_metadata(df: pd.DataFrame) -> str:
    counties = sorted(df["county"].dropna().unique().tolist())
    years = sorted(df["calendar_year"].dropna().unique().astype(int).tolist())
    cats = sorted(df["level_1_category"].dropna().unique().tolist())
    most_recent = max(years)

    # Sample rows (Suffolk, most recent year)
    sample = df[(df["county"] == "Suffolk") & (df["calendar_year"] == most_recent)].head(5)
    sample_str = sample[["calendar_year", "county", "level_1_category",
                          "level_2_category", "account_code_narrative",
                          "object_of_expenditure", "amount"]].to_string(index=False)

    return f"""=== DATA METADATA ===

CSV file: {EXP_CSV}
Shape: {len(df):,} rows × {df.shape[1]} columns

EXACT COLUMN NAMES (all lowercase after normalization):
  {list(df.columns)}

Column types:
  calendar_year         : int64   — fiscal year (e.g., 2023)
  entity_name           : object  — full entity name (e.g., "County of Suffolk")
  county                : object  — short county name (e.g., "Suffolk")
  period_start          : object  — fiscal period start date
  period_end            : object  — fiscal period end date
  account_code          : object  — budget code (e.g., "A10101")
  account_code_narrative: object  — plain-English code name
  account_code_section  : object  — always "EXPENDITURE" in this file
  level_1_category      : object  — broad spending category
  level_2_category      : object  — specific subcategory
  object_of_expenditure : object  — what money was spent on
  amount                : float64 — dollar amount (already numeric, no formatting)

All {len(counties)} NYS counties available:
  {counties}

Available calendar years: {years}
Most recent year: {most_recent}

All 14 level_1_category values:
  {cats}

Sample rows (Suffolk County, {most_recent}):
{sample_str}
"""


# ──────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ──────────────────────────────────────────────────────────────
def ask_agent(question: str, df: pd.DataFrame, collection) -> str:
    print(f"\nQuestion: {question}")
    print("Retrieving RAG context...")
    rag_context = retrieve_context(collection, question)
    data_meta = build_data_metadata(df)

    system_prompt = """You are a senior budget analyst specializing in New York State county government finance.
You have access to NYS Comptroller expenditure data and reference documentation.
Answer questions with factual analysis. Write in a professional briefing style.
Be specific — cite actual dollar amounts, percentages, and category names from the data when relevant.
If you cannot compute exact figures from the metadata provided, say so clearly and describe what analysis would be needed."""

    user_prompt = f"""{data_meta}

=== REFERENCE DOCUMENTATION (RAG) ===
{rag_context}

=== QUESTION ===
{question}

Please provide a professional, briefing-style answer. Use the data metadata and reference documentation above.
Reference exact column names when describing what analysis would be done (e.g., filter by `level_1_category`, group by `county`).
"""

    print("Sending to Claude API...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What was Suffolk County's total expenditure in the most recent year?"

    df = load_data()
    collection = get_chroma_collection()
    answer = ask_agent(question, df, collection)

    print("\n" + "=" * 60)
    print("AGENT RESPONSE")
    print("=" * 60)
    print(answer)
    print("=" * 60)
