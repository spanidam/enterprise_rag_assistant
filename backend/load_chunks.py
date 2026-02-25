import os
from langchain_core.documents import Document

CHUNKS_DIR = "../data/processed_chunks"

def load_chunk_documents():
    docs = []
    for fname in sorted(os.listdir(CHUNKS_DIR)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(CHUNKS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        # minimal metadata (we can improve later to include PDF name + page)
        docs.append(Document(page_content=text, metadata={"chunk_file": fname}))

    return docs

if __name__ == "__main__":
    docs = load_chunk_documents()
    print(f"✅ Loaded {len(docs)} chunks as Documents")
    print("Sample:", docs[0].metadata, docs[0].page_content[:150])