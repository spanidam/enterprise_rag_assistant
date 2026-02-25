from langchain_chroma import Chroma
from embeddings import STEmbeddingFunction

PERSIST_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "enterprise_rag_chunks"

def dense_search(query, k=5):
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=STEmbeddingFunction(),
        persist_directory=PERSIST_DIR
    )
    return vs.similarity_search(query, k=k)

if __name__ == "__main__":
    q = "What is transformer architecture?"
    hits = dense_search(q, k=5)
    print("QUERY:", q)
    print("="*80)
    for i, d in enumerate(hits, 1):
        print(f"\n[{i}] {d.metadata.get('chunk_file')}")
        print(d.page_content[:300])
        