"""
County Budget Agent — RAG Pipeline Builder
Run this once from the project root to embed reference docs into ChromaDB.
The vector database is stored in chroma_db/ (relative to the project root).

Usage: python build_rag.py
"""

import chromadb
import os

# --- CONFIG ---
# All paths relative to project root (where this script lives)
CHROMA_DIR = "chroma_db"
REF_DIR = "data/reference"

# --- LOAD DOCUMENTS ---
print("Loading reference documents...")

data_dict_path = os.path.join(REF_DIR, "data_dictionary.txt")
col_ref_path = os.path.join(REF_DIR, "column_reference.txt")
cat_guide_path = os.path.join(REF_DIR, "category_guide.txt")

if not os.path.exists(data_dict_path):
    raise FileNotFoundError(f"Missing: {data_dict_path}")
if not os.path.exists(col_ref_path):
    raise FileNotFoundError(f"Missing: {col_ref_path}")

with open(data_dict_path, "r", encoding="utf-8") as f:
    data_dict_raw = f.read()

with open(col_ref_path, "r", encoding="utf-8") as f:
    column_ref_raw = f.read()

category_guide_raw = ""
if os.path.exists(cat_guide_path):
    with open(cat_guide_path, "r", encoding="utf-8") as f:
        category_guide_raw = f.read()
    print(f"  category_guide.txt: {len(category_guide_raw):,} chars")

print(f"  data_dictionary.txt: {len(data_dict_raw):,} chars")
print(f"  column_reference.txt: {len(column_ref_raw):,} chars")

# --- CHUNK THE DATA DICTIONARY ---
# Split into chunks by category so each chunk is a meaningful unit.
# We split on "Category:" lines since each one groups related codes.

print("\nChunking data dictionary...")

chunks = []
chunk_ids = []
chunk_count = 0

current_section = ""
current_chunk_lines = []

for line in data_dict_raw.split("\n"):
    # New section header (EXPENDITURE, REVENUE, etc.)
    if line.startswith("--- ") and line.endswith(" ---"):
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            if len(chunk_text.strip()) > 20:
                chunks.append(chunk_text)
                chunk_ids.append(f"dict_chunk_{chunk_count}")
                chunk_count += 1
            current_chunk_lines = []
        current_section = line.strip("- ")
        current_chunk_lines.append(f"Section: {current_section}")

    # New category — start a new chunk
    elif line.startswith("Category: "):
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            if len(chunk_text.strip()) > 20:
                chunks.append(chunk_text)
                chunk_ids.append(f"dict_chunk_{chunk_count}")
                chunk_count += 1
        current_chunk_lines = [f"Section: {current_section}", line]

    # Regular line — add to current chunk
    else:
        current_chunk_lines.append(line)

# Don't forget the last chunk
if current_chunk_lines:
    chunk_text = "\n".join(current_chunk_lines)
    if len(chunk_text.strip()) > 20:
        chunks.append(chunk_text)
        chunk_ids.append(f"dict_chunk_{chunk_count}")
        chunk_count += 1

# Add column reference as its own chunk
chunks.append(column_ref_raw)
chunk_ids.append("column_reference")

# Add category guide in sections (split by "Category:" for better retrieval)
if category_guide_raw:
    cat_chunks = []
    cat_current = []
    for line in category_guide_raw.split("\n"):
        if line.startswith("Category: ") and cat_current:
            cat_chunks.append("\n".join(cat_current))
            cat_current = [line]
        else:
            cat_current.append(line)
    if cat_current:
        cat_chunks.append("\n".join(cat_current))
    for idx, cchunk in enumerate(cat_chunks):
        if len(cchunk.strip()) > 20:
            chunks.append(cchunk)
            chunk_ids.append(f"cat_guide_{idx}")
    print(f"  Added {len(cat_chunks)} category guide chunks")

# Also add an intro summary chunk for high-level queries
summary_chunk = """NYS County Budget Data Overview:
This dataset covers New York State county government financial data including expenditures and revenues.
Key columns: calendar_year, county, entity_name, level_1_category, level_2_category,
account_code_narrative, account_code_section, object_of_expenditure, amount.
Level 1 categories include: Public Safety, Education, General Government, Health,
Transportation/Highways, Social Services, Culture and Recreation, Community Services, Debt Service.
Account code sections: EXPENDITURE (spending), REVENUE (income), GL (General Ledger), FBNP (Fund Balance).
Object of expenditure types: Personal Services (salaries), Equipment and Capital Outlay, Contractual.
Data spans multiple years for all 57 NYS counties."""

chunks.append(summary_chunk)
chunk_ids.append("overview_summary")

print(f"Created {len(chunks)} total chunks")

# --- BUILD CHROMADB ---
print("\nBuilding ChromaDB vector database...")

os.makedirs(CHROMA_DIR, exist_ok=True)

# Use persistent storage so it saves to disk
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete existing collection if re-running
try:
    client.delete_collection("budget_docs")
    print("  Deleted existing 'budget_docs' collection (rebuilding fresh)")
except Exception:
    pass

# Create collection — uses ChromaDB's built-in embedding model (no API key needed)
collection = client.create_collection(
    name="budget_docs",
    metadata={"description": "NYS county budget data dictionary and column references"}
)

# Add chunks in batches
BATCH_SIZE = 50
for i in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_ids = chunk_ids[i:i + BATCH_SIZE]
    collection.add(documents=batch_chunks, ids=batch_ids)
    print(f"  Embedded batch {i // BATCH_SIZE + 1} ({len(batch_chunks)} chunks)")

print(f"\nDone! {collection.count()} chunks stored in {CHROMA_DIR}/")

# --- TEST RETRIEVAL ---
print("\n=== TESTING RAG RETRIEVAL ===\n")

test_queries = [
    "education spending",
    "public safety",
    "legislative board",
    "healthcare costs",
    "highway transportation",
]

for query in test_queries:
    results = collection.query(query_texts=[query], n_results=3)
    print(f'Query: "{query}"')
    for j, doc in enumerate(results["documents"][0]):
        preview = doc[:250].replace("\n", " ")
        print(f"  Result {j + 1}: {preview}...")
    print()

print("RAG pipeline ready. Agents can now query 'budget_docs' collection.")
