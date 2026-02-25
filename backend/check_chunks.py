import os

RAW_DOCS_PATH = "../data/raw_documents/"
CHUNKS_PATH = "../data/processed_chunks/"

raw_files = [f for f in os.listdir(RAW_DOCS_PATH) if f.endswith(".pdf")]
chunk_files = os.listdir(CHUNKS_PATH)

print(f"📄 Number of raw PDF files: {len(raw_files)}")
print(f"🧩 Number of chunk files: {len(chunk_files)}")

# Optional: print sample chunk names
print("\nSample chunks:")
for chunk in chunk_files[:5]:
    print(" -", chunk)