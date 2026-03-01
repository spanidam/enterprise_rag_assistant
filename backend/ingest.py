from pathlib import Path
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DOCS_PATH = PROJECT_ROOT / "data" / "raw_documents"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_chunks"

os.makedirs(PROCESSED_PATH, exist_ok=True)

def load_pdfs():
    documents = []
    for file in os.listdir(RAW_DOCS_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DOCS_PATH, file)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append({"filename": file, "text": text})
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "source": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })
    return chunks

def save_chunks(chunks):
    for i, chunk in enumerate(chunks):
        with open(f"{PROCESSED_PATH}/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(chunk["text"])

if __name__ == "__main__":
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    print(f"✅ Ingested {len(chunks)} chunks successfully")