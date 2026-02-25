from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever  # ✅ correct
from langchain_classic.retrievers.bm25 import BM25Retriever
from embeddings import STEmbeddingFunction
from load_chunks import load_chunk_documents

PERSIST_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "enterprise_rag_chunks"


def build_hybrid_retriever(k_dense=8, k_bm25=8, weights=(0.6, 0.4)):
    # Dense
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=STEmbeddingFunction(),
        persist_directory=PERSIST_DIR
    )
    dense = vs.as_retriever(search_kwargs={"k": k_dense})

    # BM25
    docs = load_chunk_documents()
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k_bm25

    # Ensemble fusion
    hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=list(weights))
    return hybrid

if __name__ == "__main__":
    q = "Explain retrieval augmented generation (RAG)"
    retriever = build_hybrid_retriever()
    hits = retriever.invoke(q)

    print("QUERY:", q)
    print("="*80)
    for i, d in enumerate(hits[:5], 1):
        print(f"\n[{i}] {d.metadata.get('chunk_file')}")
        print(d.page_content[:300])