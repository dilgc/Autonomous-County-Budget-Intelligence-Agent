# Autonomous County Budget Intelligence Agent

An agentic AI system that answers plain-English questions about New York State county budgets. It retrieves context from a vector store, generates and executes Python analytics code, produces charts, and writes executive briefings — all without additional user input.

Built for **AIM 490W — AI Management Capstone**, Farmingdale State College, Spring 2026.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Get an Anthropic API Key](#4-get-an-anthropic-api-key)
  - [5. Download the Data](#5-download-the-data)
  - [6. Build the Vector Store](#6-build-the-vector-store)
- [Running the Application](#running-the-application)
- [Usage Examples](#usage-examples)
- [Data](#data)
- [Known Limitations](#known-limitations)
- [Team](#team)
- [License](#license)

---

## What It Does

A user types a budget question like *"Compare public safety spending between Nassau and Suffolk from 2019 to 2024"* into a Streamlit web interface. The system then:

1. Retrieves relevant context from a ChromaDB vector store (budget category definitions, column mappings)
2. Generates Python pandas code tailored to the dataset
3. Executes that code in a sandboxed environment, generating charts when helpful
4. Synthesizes the numerical results into a 2–3 paragraph executive briefing
5. Displays everything — text, charts, data tables, and a full reasoning trace — in the browser

No additional user input is needed after the initial question.

---

## How It Works

The system follows a three-step agentic pipeline:

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│  Step 1: RAG Retrieval +        │
│          Code Generation        │
│                                 │
│  • Query ChromaDB (top-5 docs)  │
│  • Send context + metadata to   │
│    Claude                       │
│  • Claude writes pandas code    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 2: Sandboxed Execution    │
│                                 │
│  • exec() in restricted         │
│    namespace: {pd, df, plt}     │
│  • stdout captured via StringIO │
│  • Charts saved to output/      │
│  • Errors caught; auto-retry    │
│    (2 attempts max)             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 3: Narrative Synthesis    │
│                                 │
│  • Execution results + context  │
│    sent back to Claude          │
│  • Claude writes a 2–3          │
│    paragraph executive briefing │
│  • Output displayed in          │
│    Streamlit                    │
└─────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| LLM | Claude Sonnet 4 via Anthropic API |
| Vector Store | ChromaDB (local persistent, 1,575 chunks) |
| Data Manipulation | pandas |
| Visualization | matplotlib (Agg backend) |
| Web Interface | Streamlit |
| Version Control | GitHub |
| Large File Sharing | Google Drive |

---

## Project Structure

```
budget-agent/
├── .gitignore
├── .env                          # API key (not in repo)
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                      # Original Comptroller downloads
│   ├── clean/                    # Cleaned CSVs (not in repo — see below)
│   │   ├── clean_county_data.csv
│   │   ├── clean_county_expenditures.csv
│   │   ├── clean_county_revenue.csv
│   │   └── README.md             # Google Drive download link
│   └── reference/
│       ├── column_reference.txt  # Column definitions for RAG
│       └── category_guide.txt    # Query-term-to-category mappings
├── build_rag.py                  # Builds the ChromaDB vector store
├── agent_v1.py                   # Basic single-agent prototype
├── agent_v2.py                   # Full 3-step pipeline (production)
├── app.py                        # Streamlit web interface
├── chroma_db/                    # Vector store (not in repo — rebuilt by build_rag.py)
└── output/                       # Generated charts (not in repo)
```

**Files excluded from the repository** (via `.gitignore`):
- `.env` — contains the Anthropic API key
- `venv/` — Python virtual environment
- `chroma_db/` — vector store (rebuild it locally with `build_rag.py`)
- `data/clean/` — CSV files exceed GitHub's 25 MB limit (download from Google Drive)
- `output/` — generated chart images

---

## Prerequisites

- **Python 3.12** or later
- **pip** (Python package manager)
- An **Anthropic API key** (see [step 4](#4-get-an-anthropic-api-key))
- ~500 MB of disk space for the cleaned datasets

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/budget-agent.git
cd budget-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes: `anthropic`, `chromadb`, `pandas`, `matplotlib`, `streamlit`, `python-dotenv`, and their dependencies.

### 4. Get an Anthropic API Key

1. Create an account at [console.anthropic.com](https://console.anthropic.com/)
2. Generate an API key
3. Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

The application loads this key via `python-dotenv`. Never commit this file to version control.

### 5. Download the Data (optional)

The cleaned CSV files are too large for GitHub. If the CSVs are missing, the app falls back to a built-in 18-row sample dataset (Suffolk and Nassau, 2023–2024) so you can test the pipeline immediately. An amber banner and a `DEMO` badge on the dataset card make it clear when sample data is active.

For the full 57-county dataset, download the files from the Google Drive link in `data/clean/README.md` and place them in `data/clean/`:

```
data/clean/
├── clean_county_data.csv            (~29.2 MB — combined)
├── clean_county_expenditures.csv    (~13.6 MB)
└── clean_county_revenue.csv         (~8.6 MB)
```

If you want to reproduce the cleaning from scratch, run `clean_county_data.py` against the raw Comptroller ZIP files in `data/raw/`.

**Original data source:** [NYS Comptroller — Financial Data for Local Governments](https://wwe1.osc.state.ny.us/localgov/findata/financial-data-for-local-governments.cfm)

### 6. Build the Vector Store

This embeds the reference documents (column definitions, category guides) into a local ChromaDB instance:

```bash
python build_rag.py
```

This creates the `chroma_db/` directory with 1,575 embedded chunks. You only need to run this once unless you change the reference documents.

---

## Running the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

This opens a browser tab (default: `http://localhost:8501`). If the production CSVs aren't in `data/clean/`, the app starts in demo mode with a small sample dataset — you'll see an amber warning banner and a `DEMO` pill on the active dataset card. Download the full CSVs from Google Drive to switch to live data.

From there:

1. **Select a dataset** — Expenditures (91K rows), Revenues (61K rows), or Combined (224K rows)
2. **Type a question** or pick one from the example queries in the sidebar
3. **Click Analyze** — the agent retrieves context, generates code, executes it, and writes a briefing
4. **Review the output** — briefing text, charts, data tables, and an expandable reasoning trace

---

## Usage Examples

Here are queries the system handles well:

| Query Type | Example |
|------------|---------|
| Lookup | "What was Suffolk County's total expenditure in 2023?" |
| Trend | "How has education spending changed in Nassau County over the last 4 years?" |
| Comparison | "Compare public safety spending between Nassau and Suffolk in 2024" |
| Ranking | "Which spending category grew fastest in Suffolk over the last 3 years?" |
| Breakdown | "Give me a full breakdown of Suffolk's 2024 budget" |

**Verified results:**
- Suffolk County total expenditure 2023 → $4,893,403,833
- Suffolk County total expenditure 2024 → $4,642,120,061
- Nassau vs Suffolk public safety 2024 → Nassau $970.1M vs Suffolk $905.3M
- Fastest growing category (Suffolk) → Utilities +112%, General Government +25.6%

---

## Data

**Source:** [NYS Comptroller — Financial Data for Local Governments](https://wwe1.osc.state.ny.us/localgov/findata/financial-data-for-local-governments.cfm)

The cleaned expenditure file has 91,678 rows across 57 NYS counties (excluding NYC boroughs), covering 2018–2024. The revenue file has 61,011 rows with the same structure. A combined file merges both (224K rows).

**Note:** Raw CSV columns are UPPERCASE. Every script normalizes them on load with `df.columns = df.columns.str.lower()`. If you write new scripts against this data, do the same.

---

## Known Limitations

- Data covers 2018–2024 only. The five NYC boroughs are not included (they don't report through the Comptroller individually).
- Claude occasionally generates Python code with minor errors. The auto-retry mechanism handles most of these, but complex queries may sometimes need rephrasing.

---

## Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| Charles Dilger | Lead AI Engineer | Agent pipeline (`agent_v1.py`, `agent_v2.py`), tool integrations, chart generation, GitHub repo, API setup, `category_guide.txt` RAG fix |
| Hunter Dombrowski | Data Architect | Data download and cleaning, ChromaDB embedding, reference documents, `build_rag.py` |
| James Anderson | UX & Demonstration Lead | Streamlit UI (`app.py`), auto-retry mechanism, demo visuals, presentation materials |
| Michael Maurrasse | Project Manager / Consultant | Business alignment, executive-facing documentation, ethical compliance |

---

## License

This project was developed as an academic capstone for AIM 490W at Farmingdale State College. Data is sourced from public records published by the NYS Comptroller's Office.