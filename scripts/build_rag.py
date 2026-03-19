"""
County Budget Agent — RAG Pipeline Builder
Run this once to embed the data dictionary and column reference into ChromaDB.
After this runs, the vector database lives in C:\county-budget-agent\chroma_db
and is ready for the agent to search.

Usage: python build_rag.py
"""

import chromadb
import os

# --- CONFIG ---
PROJECT_DIR = r"C:\county-budget-agent"
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_db")

# --- LOAD DOCUMENTS ---
print("Loading documents...")

with open(os.path.join(PROJECT_DIR, "data_dictionary.txt"), "r") as f:
    data_dict_raw = f.read()

with open(os.path.join(PROJECT_DIR, "column_reference.txt"), "r") as f:
    column_ref_raw = f.read()

# --- CHUNK THE DATA DICTIONARY ---
# Split into chunks by category so each chunk is a meaningful unit
# We split on the "Category:" lines since each one groups related codes together

print("Chunking data dictionary...")

chunks = []
chunk_ids = []
chunk_count = 0

current_section = ""
current_chunk_lines = []

for line in data_dict_raw.split("\n"):
    # New section header (EXPENDITURE, REVENUE, etc.)
    if line.startswith("--- ") and line.endswith(" ---"):
        # Save previous chunk if it exists
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

print(f"Created {len(chunks)} chunks")

# --- BUILD CHROMADB ---
print("Building ChromaDB vector database...")

# Use persistent storage so it saves to disk
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete existing collection if re-running
try:
    client.delete_collection("budget_docs")
    print("Deleted old collection")
except:
    pass

# Create collection — uses ChromaDB's built-in embedding model (no API key needed)
collection = client.create_collection(
    name="budget_docs",
    metadata={"description": "NYS county budget data dictionary and column references"}
)

# Add chunks in batches (ChromaDB handles embedding automatically)
BATCH_SIZE = 50
for i in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_ids = chunk_ids[i:i + BATCH_SIZE]
    collection.add(documents=batch_chunks, ids=batch_ids)
    print(f"  Embedded batch {i // BATCH_SIZE + 1} ({len(batch_chunks)} chunks)")

print(f"\nDone! {collection.count()} chunks stored in {CHROMA_DIR}")

# --- TEST RETRIEVAL ---
print("\n=== TESTING RAG RETRIEVAL ===\n")

test_queries = [
    "education spending",
    "public safety police",
    "sales tax revenue",
    "healthcare health services",
    "what columns are in the data",
]

for query in test_queries:
    results = collection.query(query_texts=[query], n_results=2)
    print(f'Query: "{query}"')
    for j, doc in enumerate(results["documents"][0]):
        # Show first 200 chars of each result
        preview = doc[:200].replace("\n", " ")
        print(f"  Result {j + 1}: {preview}...")
    print()

print("RAG pipeline is ready. The agent can now search budget data context.")
