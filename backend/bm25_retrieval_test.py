from langchain_community.retrievers import BM25Retriever
from load_chunks import load_chunk_documents

def bm25_search(query, k=5):
    docs = load_chunk_documents()
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever.invoke(query)

if __name__ == "__main__":
    q = "transformer attention mechanism"
    hits = bm25_search(q, k=5)
    print("QUERY:", q)
    print("="*80)
    for i, d in enumerate(hits, 1):
        print(f"\n[{i}] {d.metadata.get('chunk_file')}")
        print(d.page_content[:300])