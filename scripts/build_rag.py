# build_rag.py
# embeds the data dictionary + column reference into chromadb
# run from project root: python scripts/build_rag.py

import chromadb
import os

# paths relative to project root, not script location
proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ref_dir = os.path.join(proj, "data", "reference")
db_path = os.path.join(proj, "chroma_db")

print("loading reference docs...")
with open(os.path.join(ref_dir, "data_dictionary.txt"), "r") as f:
    dict_raw = f.read()
with open(os.path.join(ref_dir, "column_reference.txt"), "r") as f:
    col_ref = f.read()

# chunk the dictionary by category
# each "Category:" line starts a new chunk so related codes stay together
print("chunking...")
chunks = []
ids = []
ct = 0
section = ""
buf = []

for line in dict_raw.split("\n"):
    if line.startswith("--- ") and line.endswith(" ---"):
        if buf:
            txt = "\n".join(buf)
            if len(txt.strip()) > 20:
                chunks.append(txt)
                ids.append(f"d{ct}")
                ct += 1
            buf = []
        section = line.strip("- ")
        buf.append(f"Section: {section}")

    elif line.startswith("Category: "):
        if buf:
            txt = "\n".join(buf)
            if len(txt.strip()) > 20:
                chunks.append(txt)
                ids.append(f"d{ct}")
                ct += 1
        buf = [f"Section: {section}", line]
    else:
        buf.append(line)

if buf:
    txt = "\n".join(buf)
    if len(txt.strip()) > 20:
        chunks.append(txt)
        ids.append(f"d{ct}")
        ct += 1

chunks.append(col_ref)
ids.append("col_ref")

print(f"{len(chunks)} chunks ready")

# build chromadb
print("building vector store...")
client = chromadb.PersistentClient(path=db_path)

try:
    client.delete_collection("budget_docs")
except:
    pass

coll = client.create_collection(
    name="budget_docs",
    metadata={"description": "nys county budget codes and column defs"}
)

batch = 50
for i in range(0, len(chunks), batch):
    b_chunks = chunks[i:i+batch]
    b_ids = ids[i:i+batch]
    coll.add(documents=b_chunks, ids=b_ids)
    print(f"  batch {i//batch + 1} ({len(b_chunks)} chunks)")

print(f"\n{coll.count()} chunks in {db_path}")

# quick test
print("\n--- testing retrieval ---\n")
tests = [
    "education spending",
    "public safety police",
    "sales tax revenue",
    "healthcare",
    "what columns are in the data",
]
for q in tests:
    res = coll.query(query_texts=[q], n_results=2)
    print(f'"{q}"')
    for doc in res["documents"][0]:
        preview = doc[:150].replace("\n", " ")
        print(f"  -> {preview}...")
    print()

print("done. vector store is ready.")
