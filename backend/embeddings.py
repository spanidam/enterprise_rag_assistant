from langchain_community.embeddings import HuggingFaceEmbeddings

class STEmbeddingFunction:
    """
    Bi-encoder embeddings using a strong sentence-transformers model.
    Phase 7 Optimization: Upgraded to BGE-Large for high-quality retrieval.
    """
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)