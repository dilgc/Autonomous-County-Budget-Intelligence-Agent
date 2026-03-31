"""
TASK 3 & 4: agent_v2.py — Agentic pipeline with code execution and chart generation

3-step pipeline:
  Step 1 — Code Generation: Claude writes pandas code to answer the question
  Step 2 — Code Execution: Run the code via exec(), capture output
  Step 3 — Narrative Synthesis: Claude writes an executive briefing from the results

Chart support: If a chart would help, Claude generates matplotlib code that saves
to output/chart.png. The Agg backend prevents display windows.

Usage:
    python agent_v2.py
    python agent_v2.py "Your custom question here"
"""

import os
import sys
import io
import traceback

# Fix Windows console encoding for special characters
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
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
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
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
    parts = [f"[Reference {i+1}]\n{doc[:500]}" for i, doc in enumerate(docs)]
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────
# BUILD DATA METADATA
# ──────────────────────────────────────────────────────────────
def build_data_metadata(df: pd.DataFrame) -> str:
    counties = sorted(df["county"].dropna().unique().tolist())
    years = sorted(df["calendar_year"].dropna().unique().astype(int).tolist())
    cats = sorted(df["level_1_category"].dropna().unique().tolist())
    most_recent = max(years)

    return f"""DATAFRAME AVAILABLE AS: df  (already loaded, do NOT re-read the CSV)

EXACT COLUMN NAMES (all lowercase):
  {list(df.columns)}

Column descriptions:
  calendar_year         : int64   — fiscal year
  entity_name           : object  — "County of Suffolk", etc.
  county                : object  — "Suffolk", "Nassau", etc. (57 total)
  period_start          : object  — fiscal period start date string
  period_end            : object  — fiscal period end date string
  account_code          : object  — budget code (e.g., "A10101")
  account_code_narrative: object  — plain-English code name
  account_code_section  : object  — always "EXPENDITURE" in this file
  level_1_category      : object  — broad spending category (14 values)
  level_2_category      : object  — specific subcategory
  object_of_expenditure : object  — "Personal Services", "Contractual", etc.
  amount                : float64 — dollar amount (numeric, no formatting needed)

All {len(counties)} NYS counties:
  {counties}

Available years: {years}
Most recent year: {most_recent}

All level_1_category values (14 total):
  {cats}
"""


# ──────────────────────────────────────────────────────────────
# STEP 1: CODE GENERATION
# ──────────────────────────────────────────────────────────────
def generate_code(question: str, rag_context: str, data_meta: str, client: anthropic.Anthropic) -> str:
    system_prompt = """You are a senior Python data analyst working with NYS county budget expenditure data.
Write clean, concise pandas code that answers the question.
Rules:
- The DataFrame is already loaded as `df` — do NOT read any CSV files
- Use only the exact column names provided (all lowercase)
- Store your final answer in a variable called `result`
- Use print() statements to display intermediate steps and the final answer
- Format dollar amounts with commas: f"${value:,.0f}"
- Format percentages: f"{value:.1f}%"
- If a chart would help visualize the answer, write matplotlib code.
  Save it to 'output/chart.png' using: plt.savefig('output/chart.png', dpi=150, bbox_inches='tight')
  Always call plt.close() after saving. Do NOT call plt.show().
- Output ONLY valid Python code — no markdown fences, no explanation text, just the code."""

    user_prompt = f"""=== DATA METADATA ===
{data_meta}

=== REFERENCE CONTEXT ===
{rag_context}

=== QUESTION ===
{question}

Write Python pandas code to answer this question. Remember: df is pre-loaded, use exact column names."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    code = response.content[0].text.strip()
    # Strip markdown fences if Claude added them despite instructions
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


# ──────────────────────────────────────────────────────────────
# STEP 2: CODE EXECUTION
# ──────────────────────────────────────────────────────────────
def execute_code(code: str, df: pd.DataFrame) -> tuple[str, str]:
    """Run the generated code. Returns (stdout_output, error_message)."""
    namespace = {
        "pd": pd,
        "df": df.copy(),
        "plt": plt,
        "result": None,
    }
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    error_msg = ""
    try:
        exec(code, namespace)
    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    output = stdout_capture.getvalue()
    if namespace.get("result") is not None:
        result_val = namespace["result"]
        if str(result_val) not in output:
            output += f"\nresult = {result_val}"
    return output.strip(), error_msg


# ──────────────────────────────────────────────────────────────
# STEP 3: NARRATIVE SYNTHESIS
# ──────────────────────────────────────────────────────────────
def synthesize_narrative(question: str, code: str, execution_output: str,
                          error: str, client: anthropic.Anthropic) -> str:
    if error:
        exec_summary = f"Code execution failed with error:\n{error}\n\nPartial output (if any):\n{execution_output}"
    else:
        exec_summary = execution_output or "(No output produced)"

    system_prompt = """You are a senior budget analyst writing for county government officials.
Write an executive briefing based on the data analysis results provided.
- Be direct and factual — lead with the key numbers
- Write 2-3 paragraphs in professional prose
- Cite specific dollar amounts and percentages from the execution output
- Contextualize findings within NYS county government finance norms
- If there was an execution error, note the limitation and describe what the analysis would have shown"""

    user_prompt = f"""Question asked: {question}

Analysis code that was run:
```python
{code}
```

Execution output / results:
{exec_summary}

Write a 2-3 paragraph executive briefing based on these results."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
def run_agent(question: str, df: pd.DataFrame, collection, api_client: anthropic.Anthropic):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print("=" * 60)

    # Retrieve RAG context
    rag_context = retrieve_context(collection, question)
    data_meta = build_data_metadata(df)

    # STEP 1: Generate code
    print("\n[Step 1] Generating pandas code...")
    code = generate_code(question, rag_context, data_meta, api_client)
    print("\n=== GENERATED CODE ===")
    print(code)

    # STEP 2: Execute code
    print("\n[Step 2] Executing code...")
    # Remove old chart if it exists so we can detect if a new one was created
    chart_path = os.path.join(OUTPUT_DIR, "chart.png")
    chart_existed_before = os.path.exists(chart_path)
    chart_mtime_before = os.path.getmtime(chart_path) if chart_existed_before else 0

    output, error = execute_code(code, df)

    if error:
        print(f"  WARNING: Code execution error:\n{error}")
    else:
        print("  Code executed successfully")

    print("\n=== EXECUTION RESULT ===")
    if error:
        print(f"ERROR:\n{error}")
    if output:
        print(output)
    else:
        print("(No print output)")

    # Check if chart was generated
    chart_generated = False
    chart_size = 0
    if os.path.exists(chart_path):
        chart_mtime_after = os.path.getmtime(chart_path)
        if chart_mtime_after > chart_mtime_before:
            chart_generated = True
            chart_size = os.path.getsize(chart_path)

    if chart_generated:
        print(f"\n  Chart generated: {chart_path} ({chart_size:,} bytes)")
    else:
        print("\n  No chart generated for this question")

    # STEP 3: Narrative synthesis
    print("\n[Step 3] Synthesizing executive briefing...")
    narrative = synthesize_narrative(question, code, output, error, api_client)

    print("\n=== EXECUTIVE BRIEFING ===")
    print(narrative)

    return {
        "question": question,
        "code": code,
        "output": output,
        "error": error,
        "narrative": narrative,
        "chart_generated": chart_generated,
        "chart_size": chart_size,
    }


# ──────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER (for testing / external callers)
# ──────────────────────────────────────────────────────────────
def run_pipeline(question: str) -> dict:
    """Self-contained entry point: initializes data, ChromaDB, and API client,
    then runs the full pipeline for a single question."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    api_client = anthropic.Anthropic(api_key=api_key)
    df = load_data()
    collection = get_chroma_collection()
    return run_agent(question, df, collection, api_client)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")

    api_client = anthropic.Anthropic(api_key=api_key)
    df = load_data()
    collection = get_chroma_collection()

    if len(sys.argv) > 1:
        # Single custom question
        questions = [" ".join(sys.argv[1:])]
    else:
        # Default test questions (Tasks 3 + 4)
        questions = [
            "What was Suffolk County's total expenditure in 2023?",
            "Compare public safety spending between Nassau and Suffolk in the most recent year.",
            "Which level 1 spending category grew the fastest in Suffolk County over the last 3 years?",
        ]

    for question in questions:
        result = run_agent(question, df, collection, api_client)

    print("\n" + "=" * 60)
    print("All questions processed.")
    print("=" * 60)
