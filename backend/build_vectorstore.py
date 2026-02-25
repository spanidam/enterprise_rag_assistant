import os
from langchain_chroma import Chroma
from embeddings import STEmbeddingFunction
from load_chunks import load_chunk_documents

PERSIST_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "enterprise_rag_chunks"

def build_vector_store():
    os.makedirs(PERSIST_DIR, exist_ok=True)

    docs = load_chunk_documents()
    print(f"📦 Documents to index: {len(docs)}")

    embedder = STEmbeddingFunction()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=PERSIST_DIR
    )

    # stable IDs so the same chunk_file maps to same vector
    ids = [d.metadata["chunk_file"] for d in docs]
    vector_store.add_documents(documents=docs, ids=ids)

    print(f"✅ Stored {len(docs)} chunks in Chroma at {PERSIST_DIR}")

if __name__ == "__main__":
    build_vector_store()