"""
County Budget Intelligence Agent — Streamlit Web Interface
Built by James Anderson (UX & Demonstration Lead)

Wraps the agent_v2.py backend into a polished, boardroom-ready web application.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys
import io
import re
import time
import traceback

import matplotlib
matplotlib.use("Agg")
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
REV_CSV = "data/clean/clean_county_revenue.csv"
DATA_CSV = "data/clean/clean_county_data.csv"
MODEL = "claude-sonnet-4-20250514"
RAG_TOP_K = 5
OUTPUT_DIR = "output"
MAX_RETRIES = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="County Budget Intelligence Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #fafbfc;
    }

    .block-container { padding-top: 2rem; }

    /* ── Header ── */
    .app-header {
        background: #0b1a2e;
        color: #ffffff;
        padding: 2.25rem 2.5rem;
        border-radius: 2px;
        margin-bottom: 1.75rem;
        border-bottom: 3px solid #c9a227;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 280px; height: 100%;
        background: linear-gradient(135deg, transparent 40%, rgba(201,162,39,0.06) 100%);
        pointer-events: none;
    }
    .app-header .title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0 0 0.3rem 0;
        color: #ffffff;
    }
    .app-header .subtitle {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.55);
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.02em;
    }
    .app-header .accent-line {
        display: inline-block;
        width: 36px;
        height: 2px;
        background: #c9a227;
        margin-bottom: 0.75rem;
    }

    /* ── Metrics row ── */
    .metrics-container {
        display: flex;
        gap: 0;
        margin-bottom: 1.75rem;
        border: 1px solid #dde1e6;
        border-radius: 2px;
        overflow: hidden;
        background: #ffffff;
    }
    .metric-item {
        flex: 1;
        padding: 1rem 1.25rem;
        text-align: center;
        border-right: 1px solid #dde1e6;
    }
    .metric-item:last-child { border-right: none; }
    .metric-item .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }
    .metric-item .value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0b1a2e;
        letter-spacing: -0.01em;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #0b1a2e;
        font-weight: 700;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #0b1a2e;
        display: inline-block;
    }

    /* ── Briefing output ── */
    .briefing-container {
        background: #ffffff;
        border: 1px solid #dde1e6;
        border-left: 3px solid #c9a227;
        border-radius: 2px;
        padding: 1.75rem 2rem;
        margin: 0.5rem 0 1.5rem 0;
        line-height: 1.8;
        font-size: 0.95rem;
        color: #1a1a2e;
    }

    /* ── Chart container ── */
    .chart-container {
        background: #ffffff;
        border: 1px solid #dde1e6;
        border-radius: 2px;
        padding: 1.5rem;
        margin: 0.5rem 0 1.5rem 0;
    }

    /* ── Status bar ── */
    .status-bar {
        background: #f5f6f8;
        border: 1px solid #dde1e6;
        border-radius: 2px;
        padding: 0.75rem 1.25rem;
        font-size: 0.85rem;
        color: #4b5563;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* ── Run metadata ── */
    .run-meta {
        display: flex;
        gap: 2rem;
        padding: 0.75rem 0;
        margin-top: 0.5rem;
        border-top: 1px solid #eef0f2;
        font-size: 0.78rem;
        color: #6b7280;
        font-weight: 500;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dde1e6;
    }
    .sidebar-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0b1a2e;
        letter-spacing: -0.01em;
        margin-bottom: 0.25rem;
    }
    .sidebar-divider {
        height: 1px;
        background: #dde1e6;
        margin: 1rem 0;
    }
    .sidebar-section-label {
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .sidebar-footer {
        text-align: center;
        padding: 1rem 0;
        color: #9ca3af;
        font-size: 0.72rem;
        border-top: 1px solid #eef0f2;
        margin-top: 1.5rem;
        line-height: 1.6;
    }

    /* ── Input hint ── */
    .input-hint {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Clean up Streamlit defaults ── */
    .stButton > button { border-radius: 2px; }
    div[data-testid="stExpander"] {
        border: 1px solid #dde1e6;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# UTILITY: Escape dollar signs for Streamlit markdown
# ──────────────────────────────────────────────────────────────
def escape_dollars(text: str) -> str:
    text = text.replace("$", "\\$")
    text = re.sub(r'^#{1,4}\s+.*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# SAMPLE / FALLBACK DATA
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_sample_data(section_filter: str | None = None) -> pd.DataFrame:
    """Return a small but structurally valid demo DataFrame when CSVs are missing."""
    rows = [
        # Expenditures — Suffolk
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A30101", "Police Department", "EXPENDITURE", "Public Safety", "Police", "Personal Services",     85_000_000.0),
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A30102", "Police Equipment",  "EXPENDITURE", "Public Safety", "Police", "Equipment and Capital Outlay", 4_200_000.0),
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A20101", "Public Schools",    "EXPENDITURE", "Education",     "K-12 Education", "Personal Services", 120_000_000.0),
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A10101", "Legislative Board", "EXPENDITURE", "General Government", "Administration", "Personal Services", 2_500_000.0),
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A40101", "Road Maintenance",  "EXPENDITURE", "Transportation/Highways", "Highways", "Contractual", 15_000_000.0),
        ("Suffolk", "County of Suffolk", 2024, "2024-01-01", "2024-12-31", "A30101", "Police Department", "EXPENDITURE", "Public Safety", "Police", "Personal Services",     88_500_000.0),
        ("Suffolk", "County of Suffolk", 2024, "2024-01-01", "2024-12-31", "A20101", "Public Schools",    "EXPENDITURE", "Education",     "K-12 Education", "Personal Services", 125_000_000.0),
        ("Suffolk", "County of Suffolk", 2024, "2024-01-01", "2024-12-31", "A40101", "Road Maintenance",  "EXPENDITURE", "Transportation/Highways", "Highways", "Contractual", 16_200_000.0),
        # Expenditures — Nassau
        ("Nassau", "County of Nassau",   2023, "2023-01-01", "2023-12-31", "A30101", "Police Department", "EXPENDITURE", "Public Safety", "Police", "Personal Services",     79_000_000.0),
        ("Nassau", "County of Nassau",   2023, "2023-01-01", "2023-12-31", "A20101", "Public Schools",    "EXPENDITURE", "Education",     "K-12 Education", "Personal Services", 110_000_000.0),
        ("Nassau", "County of Nassau",   2023, "2023-01-01", "2023-12-31", "A10101", "Legislative Board", "EXPENDITURE", "General Government", "Administration", "Personal Services", 2_100_000.0),
        ("Nassau", "County of Nassau",   2024, "2024-01-01", "2024-12-31", "A30101", "Police Department", "EXPENDITURE", "Public Safety", "Police", "Personal Services",     82_000_000.0),
        ("Nassau", "County of Nassau",   2024, "2024-01-01", "2024-12-31", "A20101", "Public Schools",    "EXPENDITURE", "Education",     "K-12 Education", "Personal Services", 115_000_000.0),
        # Revenue — Suffolk
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A1001",  "Property Tax",      "REVENUE", "General Government", "Tax Revenue", None,  210_000_000.0),
        ("Suffolk", "County of Suffolk", 2023, "2023-01-01", "2023-12-31", "A1002",  "State Aid",         "REVENUE", "General Government", "Intergovernmental", None, 45_000_000.0),
        ("Suffolk", "County of Suffolk", 2024, "2024-01-01", "2024-12-31", "A1001",  "Property Tax",      "REVENUE", "General Government", "Tax Revenue", None,  220_000_000.0),
        # Revenue — Nassau
        ("Nassau",  "County of Nassau",  2023, "2023-01-01", "2023-12-31", "A1001",  "Property Tax",      "REVENUE", "General Government", "Tax Revenue", None,  195_000_000.0),
        ("Nassau",  "County of Nassau",  2024, "2024-01-01", "2024-12-31", "A1001",  "Property Tax",      "REVENUE", "General Government", "Tax Revenue", None,  205_000_000.0),
    ]
    cols = ["county", "entity_name", "calendar_year", "period_start", "period_end",
            "account_code", "account_code_narrative", "account_code_section",
            "level_1_category", "level_2_category", "object_of_expenditure", "amount"]
    df = pd.DataFrame(rows, columns=cols)
    df["calendar_year"] = df["calendar_year"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    if section_filter:
        df = df[df["account_code_section"] == section_filter].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# CACHED DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_expenditure_data():
    if not os.path.exists(EXP_CSV):
        return None
    df = pd.read_csv(EXP_CSV)
    df.columns = df.columns.str.lower()
    return df

@st.cache_data(show_spinner=False)
def load_revenue_data():
    if os.path.exists(REV_CSV):
        df = pd.read_csv(REV_CSV)
        df.columns = df.columns.str.lower()
        return df
    return None

@st.cache_data(show_spinner=False)
def load_combined_data():
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        df.columns = df.columns.str.lower()
        return df
    return None

@st.cache_resource(show_spinner=False)
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("budget_docs")


# ──────────────────────────────────────────────────────────────
# AGENT PIPELINE FUNCTIONS
# ──────────────────────────────────────────────────────────────
def retrieve_context(collection, question: str, top_k: int = RAG_TOP_K) -> str:
    results = collection.query(query_texts=[question], n_results=top_k)
    docs = results["documents"][0]
    parts = [f"[Reference {i+1}]\n{doc[:500]}" for i, doc in enumerate(docs)]
    return "\n\n".join(parts)


def build_data_metadata(df: pd.DataFrame) -> str:
    counties = sorted(df["county"].dropna().unique().tolist())
    years = sorted(df["calendar_year"].dropna().unique().astype(int).tolist())
    cats = sorted(df["level_1_category"].dropna().unique().tolist())
    most_recent = max(years)
    sections = "N/A"
    if "account_code_section" in df.columns:
        sections = sorted(df["account_code_section"].dropna().unique().tolist())

    return f"""DATAFRAME AVAILABLE AS: df  (already loaded, do NOT re-read the CSV)

EXACT COLUMN NAMES (all lowercase):
  {list(df.columns)}

Column descriptions:
  calendar_year         : int64   — fiscal year
  entity_name           : object  — "County of Suffolk", etc.
  county                : object  — "Suffolk", "Nassau", etc. ({len(counties)} total)
  period_start          : object  — fiscal period start date string
  period_end            : object  — fiscal period end date string
  account_code          : object  — budget code (e.g., "A10101")
  account_code_narrative: object  — plain-English code name
  account_code_section  : object  — values: {sections}
  level_1_category      : object  — broad spending/revenue category
  level_2_category      : object  — specific subcategory
  object_of_expenditure : object  — "Personal Services", "Contractual", etc. (expenditures only)
  amount                : float64 — dollar amount (numeric, no formatting needed)

All {len(counties)} NYS counties:
  {counties}

Available years: {years}
Most recent year: {most_recent}

All level_1_category values:
  {cats}
"""


def generate_code(question, rag_context, data_meta, client):
    system_prompt = """You are a senior Python data analyst working with NYS county budget data.
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
- Make charts professional: use tight layout, clear labels, a clean style, and readable font sizes.
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
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def execute_code(code: str, df: pd.DataFrame):
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


def retry_code_generation(question, rag_context, data_meta, client, failed_code, error_msg):
    system_prompt = """You are a senior Python data analyst. The previous code had an error.
Fix it. Output ONLY valid Python code — no markdown fences, no explanation."""

    user_prompt = f"""=== DATA METADATA ===
{data_meta}

=== ORIGINAL QUESTION ===
{question}

=== FAILED CODE ===
{failed_code}

=== ERROR ===
{error_msg}

Fix the code. Remember: df is pre-loaded, all columns are lowercase. Output only Python code."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    code = response.content[0].text.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def synthesize_narrative(question, code, execution_output, error, client):
    if error:
        exec_summary = f"Code execution failed:\n{error}\n\nPartial output:\n{execution_output}"
    else:
        exec_summary = execution_output or "(No output produced)"

    system_prompt = """You are a senior budget analyst writing for county government officials.
Write an executive briefing based on the data analysis results provided.
- Be direct and factual — lead with the key numbers
- Write 2-3 paragraphs in professional prose
- Cite specific dollar amounts and percentages from the execution output
- Do NOT include any markdown headers, titles, or bold text — just write plain paragraphs
- Do NOT use markdown formatting of any kind
- If there was an error, note the limitation briefly"""

    user_prompt = f"""Question asked: {question}

Code that was run:
```python
{code}
```

Execution output:
{exec_summary}

Write a 2-3 paragraph executive briefing. No headers, no titles, no bold, no markdown — just plain paragraphs."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


def run_pipeline(question, df, collection, client, status_callback=None):
    trace = {
        "question": question,
        "rag_context": "",
        "code": "",
        "output": "",
        "error": "",
        "retries": 0,
        "narrative": "",
        "chart_generated": False,
    }

    if status_callback:
        status_callback("Searching budget reference documents...")
    rag_context = retrieve_context(collection, question)
    data_meta = build_data_metadata(df)
    trace["rag_context"] = rag_context

    if status_callback:
        status_callback("Agent is writing analysis code...")
    code = generate_code(question, rag_context, data_meta, client)
    trace["code"] = code

    chart_path = os.path.join(OUTPUT_DIR, "chart.png")
    if os.path.exists(chart_path):
        os.remove(chart_path)

    if status_callback:
        status_callback("Executing analysis...")
    output, error = execute_code(code, df)

    retries = 0
    while error and retries < MAX_RETRIES:
        retries += 1
        if status_callback:
            status_callback(f"Code error detected — auto-retrying ({retries}/{MAX_RETRIES})...")
        code = retry_code_generation(question, rag_context, data_meta, client, code, error)
        trace["code"] = code
        output, error = execute_code(code, df)

    trace["output"] = output
    trace["error"] = error
    trace["retries"] = retries
    trace["chart_generated"] = os.path.exists(chart_path)

    if status_callback:
        status_callback("Writing executive briefing...")
    narrative = synthesize_narrative(question, code, output, error, client)
    trace["narrative"] = narrative

    return trace


# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">County Budget Intelligence Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
    dataset_choice = st.radio(
        "Dataset",
        ["Expenditures", "Revenue", "Combined (All)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Sample Queries</div>', unsafe_allow_html=True)

    sample_queries = [
        "What was Suffolk County's total expenditure in 2023?",
        "Compare public safety spending between Nassau and Suffolk in 2024.",
        "Which spending category grew fastest in Suffolk over the last 3 years?",
        "Show education spending trends for Nassau County from 2018 to 2024.",
        "What are the top 5 counties by total health spending in the most recent year?",
        "Break down Suffolk County's 2024 budget by object of expenditure.",
        "How does per-category spending in Westchester compare to the state average?",
        "What percentage of Nassau's budget goes to public safety vs health?",
    ]

    for q in sample_queries:
        if st.button(q, key=f"sq_{hash(q)}", use_container_width=True):
            st.session_state["submitted_query"] = q
            st.rerun()

    st.markdown(
        '<div class="sidebar-footer">Data Source<br>NYS Comptroller\'s Office<br>Open Book New York</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <div class="accent-line"></div>
    <div class="title">County Budget Intelligence Agent</div>
    <div class="subtitle">Autonomous budget analysis for 57 New York State counties</div>
</div>
""", unsafe_allow_html=True)

# Load data
using_sample_data = False
try:
    if dataset_choice == "Expenditures":
        df = load_expenditure_data()
        dataset_label = "Expenditure"
        if df is None:
            df = build_sample_data(section_filter="EXPENDITURE")
            using_sample_data = True
    elif dataset_choice == "Revenue":
        df = load_revenue_data()
        dataset_label = "Revenue"
        if df is None:
            df = build_sample_data(section_filter="REVENUE")
            using_sample_data = True
    else:
        df = load_combined_data()
        dataset_label = "Combined"
        if df is None:
            df = build_sample_data()
            using_sample_data = True
    collection = load_chroma_collection()
except Exception as e:
    st.error(f"Failed to load data or ChromaDB: {e}")
    st.error("Make sure you've run python build_rag.py and your CSVs are in data/clean/")
    st.stop()

if using_sample_data:
    st.warning(
        "**Demo Mode — Sample Data:** The full dataset CSV was not found in `data/clean/`. "
        "A small built-in sample (2 counties, 2023–2024) is being used so you can explore "
        "the interface. Results will not reflect real budget figures. "
        "Place the production CSVs in `data/clean/` and restart to use live data.",
        icon="⚠️",
    )

# Metrics
years = sorted(df["calendar_year"].dropna().unique().astype(int).tolist())
demo_badge = ' <span style="background:#f59e0b;color:#fff;font-size:0.65rem;padding:2px 7px;border-radius:4px;vertical-align:middle;margin-left:6px;">DEMO</span>' if using_sample_data else ""
st.markdown(f"""
<div class="metrics-container">
    <div class="metric-item">
        <div class="label">Counties</div>
        <div class="value">{df['county'].nunique()}</div>
    </div>
    <div class="metric-item">
        <div class="label">Year Range</div>
        <div class="value">{min(years)} – {max(years)}</div>
    </div>
    <div class="metric-item">
        <div class="label">Records</div>
        <div class="value">{len(df):,}</div>
    </div>
    <div class="metric-item">
        <div class="label">Active Dataset</div>
        <div class="value">{dataset_label}{demo_badge}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# QUERY INPUT
# ──────────────────────────────────────────────────────────────
prefill = ""
if st.session_state.get("submitted_query"):
    prefill = st.session_state["submitted_query"]
    st.session_state["submitted_query"] = None

question = st.text_area(
    "Enter a budget question",
    value=prefill,
    height=72,
    placeholder="e.g., Compare education spending between Nassau and Suffolk from 2019 to 2024",
    label_visibility="collapsed",
)

col_btn, col_spacer = st.columns([1, 5])
with col_btn:
    run_clicked = st.button("Analyze", type="primary", use_container_width=True)

st.markdown(
    '<div class="input-hint">The agent writes Python code, executes it against live data, and produces an executive briefing.</div>',
    unsafe_allow_html=True,
)

# Trigger on button click OR sidebar prefill
should_run = run_clicked and question.strip()
if prefill.strip() and not run_clicked:
    should_run = True
    question = prefill


# ──────────────────────────────────────────────────────────────
# RUN PIPELINE
# ──────────────────────────────────────────────────────────────
if should_run:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Create a .env file with your key.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)
    status_box = st.empty()

    def update_status(msg):
        status_box.markdown(f'<div class="status-bar">{msg}</div>', unsafe_allow_html=True)

    start_time = time.time()
    with st.spinner("Agent is working..."):
        trace = run_pipeline(question, df, collection, client, status_callback=update_status)
    elapsed = time.time() - start_time
    status_box.empty()

    # ── Executive Briefing ──
    st.markdown('<div class="section-label">Executive Briefing</div>', unsafe_allow_html=True)
    st.markdown('<div class="briefing-container">', unsafe_allow_html=True)
    st.markdown(escape_dollars(trace["narrative"]))
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart ──
    chart_path = os.path.join(OUTPUT_DIR, "chart.png")
    if trace["chart_generated"] and os.path.exists(chart_path):
        st.markdown('<div class="section-label">Generated Visualization</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.image(chart_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Reasoning Trace ──
    st.markdown('<div class="section-label">Reasoning Trace</div>', unsafe_allow_html=True)
    with st.expander("View Agent Reasoning Trace", expanded=False):
        st.markdown("**Step 1 — RAG Retrieval**")
        st.text(trace["rag_context"][:1000] + ("..." if len(trace["rag_context"]) > 1000 else ""))

        st.markdown("**Step 2 — Generated Python Code**")
        st.code(trace["code"], language="python")

        st.markdown("**Step 3 — Execution Output**")
        if trace["error"]:
            st.error(f"Execution error (retried {trace['retries']}x):")
            st.code(trace["error"], language="text")
        if trace["output"]:
            st.code(trace["output"], language="text")
        else:
            st.caption("No print output")

        st.markdown("**Step 4 — Narrative Synthesis**")
        st.markdown(escape_dollars(trace["narrative"]))

    # ── Run metadata ──
    st.markdown(f"""
    <div class="run-meta">
        <span>Time: {elapsed:.1f}s</span>
        <span>Retries: {trace['retries']}</span>
        <span>Chart: {'Yes' if trace['chart_generated'] else 'No'}</span>
        <span>Model: {MODEL}</span>
    </div>
    """, unsafe_allow_html=True)

    st.session_state["history"].append({
        "question": question,
        "narrative": trace["narrative"],
        "time": elapsed,
        "chart": trace["chart_generated"],
    })

elif run_clicked and not question.strip():
    st.warning("Please enter a question.")


# ──────────────────────────────────────────────────────────────
# QUERY HISTORY
# ──────────────────────────────────────────────────────────────
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown('<div class="section-label">Query History</div>', unsafe_allow_html=True)
    for i, item in enumerate(reversed(st.session_state["history"])):
        idx = len(st.session_state["history"]) - i
        with st.expander(f"Query {idx}: {item['question'][:80]}", expanded=False):
            st.markdown(escape_dollars(item["narrative"]))
            st.caption(f"Time: {item['time']:.1f}s  |  Chart: {'Yes' if item['chart'] else 'No'}")
