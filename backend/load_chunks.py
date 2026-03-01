from pathlib import Path
from langchain_core.documents import Document

# Always resolve paths relative to THIS file, not the working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed_chunks"

def load_chunk_documents():
    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(
            f"❌ processed_chunks directory not found at: {CHUNKS_DIR}"
        )

    docs = []
    for path in sorted(CHUNKS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"chunk_file": path.name}
            )
        )

    if not docs:
        raise ValueError("❌ No .txt chunk files found in processed_chunks")

    return docs