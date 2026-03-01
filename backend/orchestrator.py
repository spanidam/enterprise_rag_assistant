from backend.hybrid_retriever import build_hybrid_retriever
from backend.reranker import CrossEncoderReranker

def retrieve_context(query, k_candidates=12, top_n=5):
    # 1) hybrid retrieval (dense + bm25)
    hybrid = build_hybrid_retriever(k_dense=k_candidates, k_bm25=k_candidates, weights=(0.6, 0.4))
    candidates = hybrid.invoke(query)

    # 2) cross-encoder rerank
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates, top_n=top_n)
    return reranked  # list of (Document, score)

if __name__ == "__main__":
    q = "What are the main components of a transformer model?"
    results = retrieve_context(q, k_candidates=12, top_n=5)

    print("QUERY:", q)
    print("="*80)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] score={float(score):.4f} source={doc.metadata.get('chunk_file')}")
        print(doc.page_content[:400])